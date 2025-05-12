#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#
import argparse
import asyncio
import os
import sys

from call_connection_manager import CallConfigManager, SessionManager
from dotenv import load_dotenv
from loguru import logger

from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import EndTaskFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.llm_service import FunctionCallParams
from pipecat.services.google.llm import GoogleLLMService
from pipecat.transports.services.daily import DailyDialinSettings, DailyParams, DailyTransport

load_dotenv(override=True)

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")

daily_api_key = os.getenv("DAILY_API_KEY", "")
daily_api_url = os.getenv("DAILY_API_URL", "https://api.daily.co/v1")


class SilenceMonitor:
    """Class to monitor silence and handle prompts or call termination."""
    
    def __init__(self, tts, llm, transport, max_silence_prompts=3, silence_threshold=10):
        self.tts = tts
        self.llm = llm
        self.transport = transport
        self.max_silence_prompts = max_silence_prompts
        self.silence_threshold = silence_threshold
        self.number_of_seconds_between_replies = 0
        self.number_of_silence_prompts = 0
        self.monitor_task = None
        self.duration = 0  # Total call duration in seconds
        self.silence_events = 0  # Number of silence events detected

    async def start_monitoring(self):
        """Start monitoring silence."""
        self.monitor_task = asyncio.create_task(self._increment_reply_timer())

    async def stop_monitoring(self):
        """Stop monitoring silence."""
        if self.monitor_task:
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except asyncio.CancelledError:
                logger.debug("Silence monitoring task cancelled.")

    async def reset_counters(self):
        """Reset silence counters when a user responds."""
        self.number_of_seconds_between_replies = 0
        self.number_of_silence_prompts = 0
        logger.info("Resetting silence counters.")

    async def _increment_reply_timer(self):
        """Increment the number_of_seconds_between_replies every second."""
        while True:
            await asyncio.sleep(1)  # Wait for 1 second
            self.number_of_seconds_between_replies += 1
            logger.debug(f"Seconds since last reply: {self.number_of_seconds_between_replies}")
            self.duration += 1

            if self.number_of_seconds_between_replies >= self.silence_threshold:
                logger.debug(f"{self.silence_threshold} seconds have passed without a reply.")
                self.number_of_seconds_between_replies = 0
                await self._handle_silence()

    async def _handle_silence(self):
        """Handle silence by sending a prompt or terminating the call."""
        self.number_of_silence_prompts += 1
        self.silence_events += 1  # Increment silence events counter
        if self.number_of_silence_prompts > self.max_silence_prompts:
            logger.debug("Exceeded maximum silence prompts. Ending the call.")
            await self.tts.say("You look busy. I will end the call. See you later.")
            self.log_summary()
            self.number_of_silence_prompts = 0
            await self.llm.queue_frame(EndTaskFrame(), FrameDirection.UPSTREAM)
           
        else:
            await self.tts.say("I am still here, waiting for your response.")
            logger.info(f"Sent silence prompt {self.number_of_silence_prompts}/{self.max_silence_prompts}.")

    def log_summary(self):
        """Log a summary of the call."""
        logger.info("Post-call summary:")
        logger.info(f"Total call duration: {self.duration} seconds")
        logger.info(f"Number of silence events: {self.silence_events}")
        logger.info(f"Number of silence prompts sent: {self.number_of_silence_prompts}")


# ------------ MAIN FUNCTION ------------

async def main(
    room_url: str,
    token: str,
    body: dict,
):
    # ------------ CONFIGURATION AND SETUP ------------

    # Create a config manager using the provided body
    call_config_manager = CallConfigManager.from_json_string(body) if body else CallConfigManager()

    # Get important configuration values
    test_mode = call_config_manager.is_test_mode()

    # Get dialin settings if present
    dialin_settings = call_config_manager.get_dialin_settings()

    # Initialize the session manager
    session_manager = SessionManager()

    # ------------ TRANSPORT SETUP ------------

    # Set up transport parameters
    if test_mode:
        logger.info("Running in test mode")
        transport_params = DailyParams(
            api_url=daily_api_url,
            api_key=daily_api_key,
            audio_in_enabled=True,
            audio_out_enabled=True,
            video_out_enabled=False,
            vad_analyzer=SileroVADAnalyzer(),
            transcription_enabled=True,
        )
    else:
        daily_dialin_settings = DailyDialinSettings(
            call_id=dialin_settings.get("call_id"), call_domain=dialin_settings.get("call_domain")
        )
        transport_params = DailyParams(
            api_url=daily_api_url,
            api_key=daily_api_key,
            dialin_settings=daily_dialin_settings,
            audio_in_enabled=True,
            audio_out_enabled=True,
            video_out_enabled=False,
            vad_analyzer=SileroVADAnalyzer(),
            transcription_enabled=True,
        )

    # Initialize transport with Daily
    transport = DailyTransport(
        room_url,
        token,
        "Simple Dial-in Bot",
        transport_params,
    )

    # Initialize TTS
    tts = CartesiaTTSService(
        api_key=os.getenv("CARTESIA_API_KEY", ""),
        voice_id="b7d50908-b17c-442d-ad8d-810c63997ed9",  # Use Helpful Woman voice by default
    )

    # ------------ FUNCTION DEFINITIONS ------------

    async def terminate_call(params: FunctionCallParams):
        """Function the bot can call to terminate the call upon completion of a voicemail message."""
        if session_manager:
            # Mark that the call was terminated by the bot
            session_manager.call_flow_state.set_call_terminated()

        # Then end the call
        await params.llm.queue_frame(EndTaskFrame(), FrameDirection.UPSTREAM)

    # Define function schemas for tools
    terminate_call_function = FunctionSchema(
        name="terminate_call",
        description="Call this function to terminate the call.",
        properties={},
        required=[],
    )

    # Create tools schema
    tools = ToolsSchema(standard_tools=[terminate_call_function])

    # ------------ LLM AND CONTEXT SETUP ------------

    # Set up the system instruction for the LLM
    system_instruction = """You are Chatbot, a friendly, helpful robot. Your goal is to demonstrate your capabilities in a succinct way. Your output will be converted to audio so don't include special characters in your answers. Respond to what the user said in a creative and helpful way, but keep your responses brief. Start by introducing yourself. If the user ends the conversation, **IMMEDIATELY** call the `terminate_call` function. """

    # Initialize LLM
    llm = GoogleLLMService(api_key=os.getenv("GOOGLE_API_KEY"))

    # Register functions with the LLM
    llm.register_function("terminate_call", terminate_call)

    # Create system message and initialize messages list
    messages = [call_config_manager.create_system_message(system_instruction)]

    # Initialize LLM context and aggregator
    context = OpenAILLMContext(messages, tools)
    context_aggregator = llm.create_context_aggregator(context)

    # ------------ PIPELINE SETUP ------------

    # Build pipeline
    pipeline = Pipeline(
        [
            transport.input(),  # Transport user input
            context_aggregator.user(),  # User responses
            llm,  # LLM
            tts,  # TTS
            transport.output(),  # Transport bot output
            context_aggregator.assistant(),  # Assistant spoken responses
        ]
    )

    # Create pipeline task
    task = PipelineTask(pipeline, params=PipelineParams(allow_interruptions=True))

    # Initialize SilenceMonitor
    silence_monitor = SilenceMonitor(tts, llm, transport)

    # ------------ EVENT HANDLERS ------------

    @transport.event_handler("on_first_participant_joined")
    async def on_first_participant_joined(transport, participant):
        logger.debug(f"First participant joined: {participant['id']}")
        await transport.capture_participant_transcription(participant["id"])
        await task.queue_frames([context_aggregator.user().get_context_frame()])
        # Start the silence monitor
        await silence_monitor.start_monitoring()

    @transport.event_handler("on_participant_left")
    async def on_participant_left(transport, participant, reason):
        logger.debug(f"Participant left: {participant}, reason: {reason}")
        await task.cancel()
        await silence_monitor.stop_monitoring()
        silence_monitor.log_summary()  # Log the post-call summary when the user leaves

    @transport.event_handler("on_transcription_message")
    async def on_transcription_message(transport, transcription):
        logger.info(f"Customer message: {transcription}")
        await silence_monitor.reset_counters()

    # ------------ RUN PIPELINE ------------

    if test_mode:
        logger.debug("Running in test mode (can be tested in Daily Prebuilt)")

    runner = PipelineRunner()
    await runner.run(task)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple Dial-in Bot")
    parser.add_argument("-u", "--url", type=str, help="Room URL")
    parser.add_argument("-t", "--token", type=str, help="Room Token")
    parser.add_argument("-b", "--body", type=str, help="JSON configuration string")

    args = parser.parse_args()

    # Log the arguments for debugging
    logger.info(f"Room URL: {args.url}")
    logger.info(f"Token: {args.token}")
    logger.info(f"Body provided: {bool(args.body)}")

    asyncio.run(main(args.url, args.token, args.body))
