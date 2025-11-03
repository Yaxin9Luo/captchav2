"""
Simple CLI to drive the OpenCaptchaWorld benchmark with a browser-use agent.

"""

from __future__ import annotations

import argparse
import asyncio
import logging
import textwrap
from typing import Callable, Optional

from browser_use import Agent, Browser


def _build_task_prompt(url: str, limit: int) -> str:
	"""
	Create the instruction string passed to the browser-use agent.

	Args:
	    url: Fully qualified URL pointing at a running OpenCaptchaWorld instance.
	    limit: Number of puzzles the agent should attempt before finishing.
	"""
	instructions = f"""
	You are evaluating Open CaptchaWorld puzzles hosted at {url}.
	You have VISION CAPABILITIES - you can see images, animations, and all visual content on the page.
	Use your vision to carefully analyze and solve each puzzle.

	AVAILABLE TOOLS:
	- VISION: You can see the entire page visually - use this to analyze puzzles visually and locate elements

	CRITICAL MEMORY INSTRUCTION:
	- Treat EACH puzzle as COMPLETELY NEW and INDEPENDENT.
	- DO NOT rely on information, patterns, or assumptions from previous puzzles.
	- DO NOT assume the current puzzle is similar to the previous one.
	- When a new puzzle loads, CLEAR your mental state and approach it fresh.
	- Always read the CURRENT puzzle prompt/question text carefully - don't assume it's the same as before.
	- Always examine the CURRENT visual content - don't rely on what you saw in previous puzzles.
	- If you see the same puzzle type twice, treat them as completely different instances.

	CRITICAL WORKFLOW - Follow these steps EXACTLY for each puzzle:
	
	For EACH puzzle (treat as NEW):
	1. CAREFULLY EXAMINE the visual content on the page using your vision:
	   - Look at ALL images, dice, shapes, animations, or visual elements displayed RIGHT NOW
	   - Read ALL text on the page, especially the puzzle prompt/question - read it fresh, don't assume
	   - Identify the puzzle type (Dice Count, Shadow, Mirror, etc.) by looking at what's ACTUALLY displayed
	   
	2. ANALYZE the puzzle based on what you SEE RIGHT NOW:
	   - For Dice Count: Look at EVERY die shown, identify the number on TOP of each die, add ALL numbers together
	   - For Shadow Plausible: Examine each image's shadow and determine if it's physically plausible
	   - For Mirror puzzles: Compare reference image with mirror options visually
	   - For Red Dot: This is TIME-SENSITIVE! You must click the red dot element IMMEDIATELY after seeing it.
	     * STEP 1: Use your VISION to visually see the red circular dot on the page
	     * STEP 2: Click the red dot element directly once located - no submit button needed
	     * The red dot appears as a small red circular element, usually in a dashed border area
	     * The dot disappears after 10 seconds, so act FAST when you see it
	     * If you see "(X/5)" in the prompt, you need to click 5 dots in sequence - click each one as soon as it appears
	     * PREFERRED METHOD: Use vision to see it, find its location in the image then click using its index or selector "#dot"
	   - For other types: Use visual analysis appropriate to that puzzle type based on CURRENT content
	   
	3. CALCULATE or DETERMINE your answer carefully based on CURRENT visual data:
	   - Count accurately, don't guess
	   - Double-check your calculation if it's a math problem
	   - Make sure you've examined ALL visual elements CURRENTLY displayed before answering
	   
	4. MAKE YOUR SELECTION or ENTER YOUR ANSWER:
	   - For multiple choice puzzles: Click on the correct answer/option/image
	   - For input-based puzzles: Type your answer into the input field
	   - For interactive puzzles: Perform the required interaction (e.g., clicking images, selecting options)
	   
	5. CRITICAL - SUBMIT YOUR ANSWER:
	   - MOST puzzles require clicking a SUBMIT button after making your selection
	   - Look for a "Submit" button, "Submit Answer" button, or similar submission element
	   - If you see a submit button, you MUST click it - your answer is NOT submitted until you click submit
	   - ONLY Red Dot puzzles don't have a submit button (clicking the dot itself submits)
	   - If you made a selection/entered an answer but didn't click submit, the answer hasn't been submitted yet
	   - DO NOT check results until AFTER you have clicked the submit button (if one exists)
	   
	6. WAIT 1 second after clicking submit for the result message to appear in #result-message.
	7. Check the result message ONCE - it will say "Correct!" or "Incorrect." or show an error.
	8. DO NOT repeatedly check the result message - check it once and move on.
	9. DETECTING NEW PUZZLE: A new puzzle has loaded when:
	   - The prompt text changes to something different
	   - The puzzle type indicator changes (if visible)
	   - Visual content changes significantly (new images, different layout)
	   - For Red Dot: The counter resets (e.g., goes from "(5/5)" or "(X/5)" back to "(0/5)" or new prompt appears)
	10. After seeing the result, IMMEDIATELY FORGET everything about that puzzle and prepare for the next one.
	11. If the SAME prompt text appears for more than 5 seconds after a failure, the puzzle likely hasn't changed yet.
	    In this case, wait 2 more seconds, then check if anything changed. If still the same, proceed as if it's a new puzzle attempt.
	12. When you see NEW puzzle content (different prompt, different images, or UI changes), treat it as a BRAND NEW puzzle - start from step 1.
	13. If no new puzzle appears after 5 seconds total, check the page state - it may be loading or stuck.
	
	IMPORTANT: VISUAL ANALYSIS TIPS:
	- Zoom in or scroll if needed to see details clearly
	- Make sure you've seen ALL dice/images/elements CURRENTLY displayed before answering
	- For counting puzzles, count systematically: go through each item one by one as it appears NOW
	- Read numbers on dice carefully - look at the top face of each die that's ACTUALLY visible
	- If unsure, examine the visual content more carefully before submitting - look at what's ACTUALLY there

	
	ANTI-LOOP RULES:
	- NEVER check the same element more than 2 times in a row.
	- If an evaluate() call returns the same result twice, MOVE ON.
	- CRITICAL: After making a selection/entering an answer, you MUST click the SUBMIT button before checking results (except Red Dot puzzles)
	- After clicking submit, wait 1-2 seconds before checking results.
	- After checking results, immediately look for the next puzzle and RESET your approach.
	- If stuck on the same action for 3+ steps, try a different approach or skip to the next puzzle.
	- If you get "Incorrect" multiple times on the same puzzle, re-examine the CURRENT visual content more carefully
	- CRITICAL: If you see the EXACT SAME prompt text for 4+ consecutive steps, the puzzle hasn't changed - wait 3 seconds then treat as new
	- For Red Dot: If you fail, wait for the counter to reset or prompt to change before trying again - don't keep clicking if you already failed
	- NEVER check results before clicking submit - if you see a submit button, you must click it first
	
	STOPPING CONDITIONS:
	- Stop after attempting {limit} puzzles (count both correct and incorrect).
	- If you cannot proceed (page unresponsive, no puzzles loading), summarize what you completed and stop.
	
	IMPORTANT CONSTRAINTS:
	- Stay ONLY on the captcha page - do not navigate to other websites.
	- Do not access local files or directories.
	- Do not refresh the page unless absolutely necessary (page completely frozen).

	At the end, provide a summary listing each puzzle type attempted and whether it was solved successfully.
	"""
	return textwrap.dedent(instructions).strip()


def _resolve_browser_use_client():
	"""
	Return ChatBrowserUse class, falling back to internal path for older releases.
	"""
	try:
		from browser_use import ChatBrowserUse  # type: ignore

		return ChatBrowserUse
	except ImportError as exc:  # Attribute missing or module not exposing class
		try:
			from browser_use.llm.browser_use.chat import ChatBrowserUse  # type: ignore

			return ChatBrowserUse
		except ImportError as inner_exc:
			raise ImportError(
				'ChatBrowserUse is not available in the installed browser-use package. '
				'Upgrade to the latest release (`pip install -U "browser-use[cli]"`).'
			) from inner_exc
	except Exception as exc:  # pragma: no cover - defensive
		raise ImportError('Failed to load ChatBrowserUse client') from exc


def _create_llm_factory() -> dict[str, Callable[[argparse.Namespace], object]]:
	"""Return a mapping from CLI option to LLM constructor callables."""

	def browser_use_factory(args: argparse.Namespace):
		if args.model:
			raise ValueError(
				'The browser-use LLM does not support custom model names via --model. '
				'Use --fast flag instead to use the fast ChatBrowserUse model, or omit --model.'
			)
		try:
			ChatBrowserUse = _resolve_browser_use_client()
		except ImportError as exc:
			raise ValueError(str(exc)) from exc
		return ChatBrowserUse(fast=args.fast)

	def openai_factory(args: argparse.Namespace):
		from browser_use import ChatOpenAI

		model = args.model or 'gpt-4.1-mini'
		try:
			return ChatOpenAI(model=model)
		except TypeError as exc:
			raise ValueError(f'Invalid OpenAI configuration: {exc}') from exc

	def anthropic_factory(args: argparse.Namespace):
		from browser_use import ChatAnthropic

		model = args.model or 'claude-3-7-sonnet-20250219'
		try:
			return ChatAnthropic(model=model)
		except TypeError as exc:
			raise ValueError(f'Invalid Anthropic configuration: {exc}') from exc

	def google_factory(args: argparse.Namespace):
		from browser_use import ChatGoogle

		model = args.model or 'gemini-2.0-flash'
		try:
			return ChatGoogle(model=model)
		except TypeError as exc:
			raise ValueError(f'Invalid Google configuration: {exc}') from exc

	def groq_factory(args: argparse.Namespace):
		from browser_use import ChatGroq

		model = args.model or 'llama-3.1-70b-versatile'
		try:
			return ChatGroq(model=model)
		except TypeError as exc:
			raise ValueError(f'Invalid Groq configuration: {exc}') from exc

	def azure_factory(args: argparse.Namespace):
		from browser_use import ChatAzureOpenAI

		model = args.model
		if not model:
			raise ValueError('--model is required when using azure-openai (pass via --model)')
		try:
			return ChatAzureOpenAI(model=model)
		except TypeError as exc:
			raise ValueError(f'Invalid Azure OpenAI configuration: {exc}') from exc

	return {
		'browser-use': browser_use_factory,
		'openai': openai_factory,
		'anthropic': anthropic_factory,
		'google': google_factory,
		'groq': groq_factory,
		'azure-openai': azure_factory,
	}


def _configure_logging(verbose: bool) -> None:
	"""Set a minimal logging format for the CLI."""
	level = logging.DEBUG if verbose else logging.INFO
	logging.basicConfig(level=level, format='[%(levelname)s] %(message)s')


def _create_browser(args: argparse.Namespace) -> Browser | None:
	"""
	Create a Browser session if custom configuration is required.

	If no special flags are set, returning None lets browser-use create the default session.
	"""
	browser_kwargs: dict[str, object] = {}
	if args.use_cloud:
		browser_kwargs['use_cloud'] = True
	if args.headless:
		browser_kwargs['headless'] = True
	if args.window_width or args.window_height:
		width = args.window_width or 1280
		height = args.window_height or 720
		browser_kwargs['viewport'] = {'width': width, 'height': height}
		browser_kwargs['window_size'] = {'width': width, 'height': height}

	if not browser_kwargs:
		return None

	return Browser(**browser_kwargs)


def _get_model_info(args: argparse.Namespace, llm_name: str) -> str:
	"""Return a human-readable string describing which model is being used."""
	if llm_name == 'browser-use':
		model_desc = 'ChatBrowserUse (fast)' if args.fast else 'ChatBrowserUse (standard)'
	else:
		# For other LLMs, show the model name or default
		defaults = {
			'openai': 'gpt-4.1-mini',
			'anthropic': 'claude-3-7-sonnet-20250219',
			'google': 'gemini-2.0-flash',
			'groq': 'llama-3.1-70b-versatile',
			'azure-openai': None,  # Required, no default
		}
		model = args.model or defaults.get(llm_name, 'default')
		if model:
			model_desc = model
		else:
			model_desc = 'default'
	return f'{llm_name} ({model_desc})'


async def _run_agent(args: argparse.Namespace) -> int:
	"""Run the browser-use agent with the provided CLI options."""
	llm_factories = _create_llm_factory()
	llm_name = args.llm.lower()

	if llm_name not in llm_factories:
		choices = ', '.join(sorted(llm_factories))
		raise ValueError(f'Unsupported llm "{args.llm}". Choose from: {choices}')

	llm = llm_factories[llm_name](args)
	
	# Try to get actual model info from the LLM object if available
	actual_model = None
	try:
		# Try common attribute names for model info
		actual_model = (
			getattr(llm, 'model_name', None) or
			getattr(llm, 'model', None) or
			getattr(llm, '_model_name', None) or
			getattr(llm, 'llm', None) or  # Some wrappers have nested LLM
			getattr(llm, '_llm', None)
		)
		# If we got a nested LLM object, try to get model from it
		if actual_model and hasattr(actual_model, 'model_name'):
			actual_model = actual_model.model_name
		elif actual_model and hasattr(actual_model, 'model'):
			actual_model = actual_model.model
		# For browser-use, try to inspect deeper
		if llm_name == 'browser-use':
			fast_mode = getattr(llm, 'fast', None)
			# Try to find the underlying model by checking nested objects
			if hasattr(llm, 'llm') or hasattr(llm, '_llm'):
				nested = getattr(llm, 'llm', None) or getattr(llm, '_llm', None)
				if nested:
					nested_model = (
						getattr(nested, 'model_name', None) or
						getattr(nested, 'model', None) or
						getattr(nested, '_model_name', None)
					)
					if nested_model:
						actual_model = f'{nested_model} (fast={fast_mode})' if fast_mode is not None else nested_model
			if not actual_model and fast_mode is not None:
				actual_model = f'ChatBrowserUse (fast={fast_mode})'
	except Exception as e:
		if args.verbose:
			logging.debug('Could not extract model info: %s', e)
	
	model_info = _get_model_info(args, llm_name)
	if actual_model:
		logging.info('Using LLM: %s (actual model: %s)', model_info, actual_model)
	else:
		logging.info('Using LLM: %s', model_info)
	browser = _create_browser(args)
	task = _build_task_prompt(args.url, args.limit)

	# Set max_failures to prevent infinite loops - agent will stop after consecutive failures
	# Limit memory/history to prevent the model from carrying over context between puzzles
	agent_kwargs = {
		'max_failures': args.max_failures,
		'max_history_items': 6,  # Limit conversation history to prevent memory buildup
		'include_recent_events': False,  # Don't include recent events to reduce context
		'use_thinking': True,  # Disable thinking mode to reduce memory retention
	}
	agent = Agent(
		task=task,
		llm=llm,
		browser=browser,
		max_actions_per_step=args.max_actions_per_step,
		include_tool_call_examples=False,
		**agent_kwargs,
	)

	if args.verbose:
		logging.getLogger('browser_use').setLevel(logging.DEBUG)

	try:
		history = await agent.run(max_steps=args.max_steps)
	except ValueError as exc:
		# Commonly raised when the chosen LLM requires API keys.
		logging.error(str(exc))
		return 1

	final_text = history.final_result()
	successful = history.is_successful()
	logging.info('Agent finished. success=%s', successful)
	if final_text:
		print('\nFinal summary:\n')
		print(final_text)
	else:
		print('\nAgent did not produce a final summary. Inspect history for details.')

	if history.usage and history.usage.total_cost is not None:
		logging.info('Approximate token cost: %s', history.usage.total_cost)

	return 0 if successful is not False else 1


def _build_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description='Run a browser-use agent on Open CaptchaWorld puzzles.')
	llm_choices = sorted(_create_llm_factory().keys())
	parser.add_argument('--url', default='http://127.0.0.1:7860', help='URL of the running OpenCaptchaWorld instance.')
	parser.add_argument('--limit', type=int, default=265, help='Number of puzzle attempts before the agent stops.')
	parser.add_argument(
		'--llm',
		choices=llm_choices,
		default='browser-use',
		help='LLM backend to use.',
	)
	parser.add_argument(
		'--model',
		help='Override the model name for the selected LLM (if supported). '
		'Note: browser-use backend uses fixed internal models and does not support --model.'
	)
	parser.add_argument(
		'--fast',
		action='store_true',
		help='Use the fast ChatBrowserUse model when --llm browser-use. '
		'Standard mode uses a more powerful model; fast mode uses a faster/cheaper model. '
		'Both are fixed models managed by the browser-use library.'
	)
	parser.add_argument('--max-steps', type=int, default=60, help='Maximum agent reasoning steps.')
	parser.add_argument(
		'--max-actions-per-step',
		type=int,
		default=10,
		help='Limit number of browser actions the agent may take per reasoning step.',
	)
	parser.add_argument(
		'--max-failures', 
		type=int, 
		default=20, 
		help='Consecutive failure limit before aborting. Lower values prevent infinite loops. (default: 20)'
	)
	parser.add_argument('--use-cloud', action='store_true', help='Launch the browser in the Browser Use Cloud.')
	parser.add_argument('--headless', action='store_true', help='Run the local browser in headless mode.')
	parser.add_argument('--window-width', type=int, help='Browser viewport width (pixels).')
	parser.add_argument('--window-height', type=int, help='Browser viewport height (pixels).')
	parser.add_argument('--verbose', action='store_true', help='Enable verbose logging.')
	return parser


def main(argv: Optional[list[str]] = None) -> int:
	parser = _build_parser()
	args = parser.parse_args(argv)
	_configure_logging(args.verbose)
	try:
		return asyncio.run(_run_agent(args))
	except KeyboardInterrupt:
		print('\nInterrupted by user.')
		return 1
	except ValueError as exc:
		# User input errors - show clean message without traceback
		logging.error('%s', exc)
		return 1
	except Exception as exc:  # pylint: disable=broad-except
		# Unexpected errors - show full traceback
		logging.exception('Agent run failed: %s', exc)
		return 1


if __name__ == '__main__':
	raise SystemExit(main())
