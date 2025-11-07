"""
Simple CLI to drive the OpenCaptchaWorld benchmark with a browser-use agent.

"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import textwrap
import urllib.error
import urllib.request
from urllib.parse import urljoin
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
	- Never refresh the page!!!
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


def _post_agent_metadata_to_server(base_url: str, metadata: dict[str, str]) -> None:
	"""Notify the benchmark server about the active agent metadata."""
	if not base_url:
		return

	try:
		endpoint = urljoin(base_url if base_url.endswith('/') else base_url + '/', 'api/agent_metadata')
	except Exception as exc:  # pragma: no cover - defensive
		logging.debug('Could not build agent metadata endpoint: %s', exc)
		return

	try:
		request_body = json.dumps(metadata).encode('utf-8')
		req = urllib.request.Request(
			endpoint,
			data=request_body,
			headers={'Content-Type': 'application/json'},
		)
		with urllib.request.urlopen(req, timeout=5) as resp:
			status = getattr(resp, 'status', None)
			if status and status >= 400:
				logging.debug('Agent metadata registration returned HTTP %s', status)
			else:
				logging.debug('Registered agent metadata with server (%s)', metadata)
	except urllib.error.URLError as exc:
		logging.debug('Could not send agent metadata to server: %s', exc)
	except Exception as exc:  # pragma: no cover - defensive
		logging.debug('Unexpected error when sending agent metadata: %s', exc)


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
	
	# Extract model name and provider for metadata injection
	# Get model name
	model_name = args.model if args.model else None
	if not model_name:
		if llm_name == 'browser-use':
			model_name = 'ChatBrowserUse (fast)' if args.fast else 'ChatBrowserUse (standard)'
		else:
			defaults = {
				'openai': 'gpt-4.1-mini',
				'anthropic': 'claude-3-7-sonnet-20250219',
				'google': 'gemini-2.0-flash',
				'groq': 'llama-3.1-70b-versatile',
				'azure-openai': None,
			}
			model_name = defaults.get(llm_name, 'default')
		if actual_model:
			# Use actual model if available
			if isinstance(actual_model, str):
				model_name = actual_model.split('(')[0].strip() if '(' in actual_model else actual_model
	
	# Get provider name - use lowercase with hyphen to match browser-use library format
	provider_name = llm_name.lower()
	if llm_name == 'browser-use':
		provider_name = 'browser-use'  # Keep as-is to match library format
	elif llm_name == 'azure-openai':
		provider_name = 'azure-openai'
	
	# Log initial metadata values for debugging
	logging.info(f'Initial metadata - Model: {model_name}, Provider: {provider_name}, Framework: browser-use')
	
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

	# Try to extract actual model name from agent after creation
	# The browser-use library logs the model name during Agent initialization
	# Try to get it from agent.llm or agent's internal state
	if llm_name == 'browser-use':
		try:
			# Try to get model from agent's llm attribute
			agent_llm = getattr(agent, 'llm', None) or llm
			# Try various ways to get the actual model name
			extracted_model = None
			
			# Method 1: Check agent.llm directly
			if agent_llm:
				extracted_model = (
					getattr(agent_llm, 'model_name', None) or
					getattr(agent_llm, 'model', None) or
					getattr(agent_llm, '_model_name', None)
				)
			
			# Method 2: Check nested llm objects and their __dict__
			if not extracted_model and agent_llm:
				for attr_name in ['llm', '_llm', 'client', '_client', '_chat_model']:
					nested = getattr(agent_llm, attr_name, None)
					if nested:
						# Try direct attributes
						nested_model = (
							getattr(nested, 'model_name', None) or
							getattr(nested, 'model', None) or
							getattr(nested, '_model_name', None) or
							getattr(nested, '_model', None)
						)
						if nested_model:
							extracted_model = nested_model
							break
						# Try checking __dict__ for model-related keys
						if hasattr(nested, '__dict__'):
							for key in nested.__dict__.keys():
								if 'model' in key.lower():
									value = getattr(nested, key, None)
									if isinstance(value, str) and value:
										extracted_model = value
										break
							if extracted_model:
								break
			
			# Method 3: Try to get from agent's internal state and __dict__
			if not extracted_model:
				for attr_name in ['_llm', '_model', '_model_name']:
					agent_model = getattr(agent, attr_name, None)
					if agent_model:
						if isinstance(agent_model, str):
							extracted_model = agent_model
						else:
							extracted_model = (
								getattr(agent_model, 'model_name', None) or
								getattr(agent_model, 'model', None) or
								getattr(agent_model, '_model_name', None) or
								getattr(agent_model, '_model', None)
							)
						if extracted_model:
							break
				
				# Check agent's __dict__ for model-related attributes
				if not extracted_model and hasattr(agent, '__dict__'):
					for key in agent.__dict__.keys():
						if 'model' in key.lower():
							value = getattr(agent, key, None)
							if isinstance(value, str) and value:
								extracted_model = value
								break
			
			# Method 4: Deep inspection of agent_llm's __dict__
			if not extracted_model and agent_llm and hasattr(agent_llm, '__dict__'):
				for key, value in agent_llm.__dict__.items():
					if 'model' in key.lower() and isinstance(value, str) and value:
						extracted_model = value
						break
			
			# If we found a model name, use it (clean it up if needed)
			if extracted_model:
				if isinstance(extracted_model, str):
					# Clean up the model name - remove any extra info in parentheses
					clean_model = extracted_model.split('(')[0].strip() if '(' in extracted_model else extracted_model.strip()
					# Remove quotes if present
					clean_model = clean_model.strip('"\'')
					if clean_model and clean_model != model_name:
						model_name = clean_model
						logging.info(f'Extracted actual model name from agent: {model_name}')
			else:
				# Fallback: Use "bu-1-0" for standard mode based on browser-use library logs
				# The library logs show "model=bu-1-0" for the standard model
				if not args.fast:
					model_name = 'bu-1-0'
					logging.info(f'Using fallback model name for browser-use standard mode: {model_name}')
				# If we couldn't extract, log what we tried for debugging
				if args.verbose:
					logging.debug(f'Could not extract model name. agent.llm type: {type(agent_llm)}, agent type: {type(agent)}')
		except Exception as e:
			if args.verbose:
				logging.debug('Could not extract model name from agent: %s', e)
	
	# Log final metadata values that will be injected
	logging.info(f'Final metadata to inject - Model: {model_name}, Provider: {provider_name}, Framework: browser-use')

	# Notify the benchmark server of the current metadata so it can enrich results
	_post_agent_metadata_to_server(
		args.url,
		{
			'model': model_name,
			'provider': provider_name,
			'agent_framework': 'browser-use',
			'agentFramework': 'browser-use',
		},
	)

	if args.verbose:
		logging.getLogger('browser_use').setLevel(logging.DEBUG)

	# Track cost incrementally
	previous_cost = 0.0
	puzzle_count = 0
	
	# Inject cost tracking and metadata script into the browser page
	# This will allow JavaScript to access cost data and model/provider info
	# We use localStorage to persist metadata across page reloads
	# NOTE: This script is created AFTER model extraction, so it uses the updated model_name
	metadata_script = f"""
	(function() {{
		// Store metadata in localStorage for persistence across page reloads
		const METADATA = {json.dumps({"model": model_name, "provider": provider_name, "agentFramework": "browser-use"})};
		try {{
			if (typeof localStorage !== 'undefined') {{
				localStorage.setItem('__agentMetadata', JSON.stringify(METADATA));
				console.log('Stored agent metadata in localStorage:', METADATA);
			}} else {{
				console.warn('localStorage not available, metadata will not persist across page reloads');
			}}
		}} catch(e) {{
			console.warn('Could not store metadata in localStorage:', e);
		}}
		
		function injectMetadata() {{
			// First try to get from localStorage
			try {{
				const stored = localStorage.getItem('__agentMetadata');
				if (stored) {{
					window.__agentMetadata = JSON.parse(stored);
				}} else {{
					window.__agentMetadata = METADATA;
				}}
			}} catch(e) {{
				window.__agentMetadata = METADATA;
			}}
			
			if (!window.__agentCostTracker) {{
				window.__agentCostTracker = {{
					costs: [],
					totalCost: 0,
					puzzleCount: 0,
					addCost: function(cost) {{
						this.costs.push(cost);
						this.totalCost += cost;
						this.puzzleCount += 1;
					}},
					getAverageCost: function() {{
						return this.puzzleCount > 0 ? this.totalCost / this.puzzleCount : 0;
					}},
					getCurrentCost: function() {{
						return this.totalCost;
					}}
				}};
			}}
		}}
		
		// Inject immediately
		injectMetadata();
		
		// Re-inject on page load (for SPA navigation)
		if (document.readyState === 'loading') {{
			document.addEventListener('DOMContentLoaded', injectMetadata);
		}} else {{
			injectMetadata();
		}}
		
		// Also re-inject periodically to ensure it persists (every 2 seconds)
		setInterval(injectMetadata, 2000);
		
		// Inject as a script tag in head for persistence across page reloads
		const scriptId = '__agent_metadata_injector';
		let existingScript = document.getElementById(scriptId);
		if (existingScript) {{
			existingScript.remove();
		}}
		const scriptTag = document.createElement('script');
		scriptTag.id = scriptId;
		const metadataJson = JSON.stringify(METADATA);
		scriptTag.textContent = `(function(){{try{{const s=localStorage.getItem('__agentMetadata');window.__agentMetadata=s?JSON.parse(s):{json.dumps({"model": model_name, "provider": provider_name, "agentFramework": "browser-use"})};}}catch(e){{window.__agentMetadata={json.dumps({"model": model_name, "provider": provider_name, "agentFramework": "browser-use"})};}}function i(){{try{{const s=localStorage.getItem('__agentMetadata');if(s){{window.__agentMetadata=JSON.parse(s);}}}}catch(e){{}}window.__agentMetadata=window.__agentMetadata||{json.dumps({"model": model_name, "provider": provider_name, "agentFramework": "browser-use"})};}}i();if(document.readyState==='loading'){{document.addEventListener('DOMContentLoaded',i);}}setInterval(i,2000);}})();`;
		if (document.head) {{
			document.head.appendChild(scriptTag);
		}} else {{
			document.addEventListener('DOMContentLoaded', function() {{
				document.head.appendChild(scriptTag);
			}});
		}}
	}})();
	"""
	
	try:
		# Initialize cost tracking and metadata in browser before running agent
		# Access browser from agent or use the browser instance we created
		browser_instance = browser if browser else (getattr(agent, 'browser', None) if hasattr(agent, 'browser') else None)
		if browser_instance:
			try:
				await browser_instance.execute_script(metadata_script)
				if args.verbose:
					logging.info('Injected metadata script before agent run')
			except Exception as e:
				if args.verbose:
					logging.debug('Could not inject metadata/cost tracking script: %s', e)
		
		# Also inject after delays to catch post-navigation
		async def delayed_injection():
			# Inject multiple times at different intervals to ensure it sticks
			delays = [2, 5, 10, 20]
			for i, delay in enumerate(delays):
				if i == 0:
					await asyncio.sleep(delay)
				else:
					await asyncio.sleep(delay - delays[i-1])  # Sleep for the difference
				if browser_instance:
					try:
						await browser_instance.execute_script(metadata_script)
						# Verify it was stored
						verification = await browser_instance.evaluate("""
							(() => {
								try {
									const stored = localStorage.getItem('__agentMetadata');
									return stored ? 'stored' : 'not stored';
								} catch(e) {
									return 'error: ' + e.message;
								}
							})()
						""")
						if args.verbose:
							logging.info(f'Injected metadata after {delay}s delay. Verification: {verification}')
					except Exception as e:
						if args.verbose:
							logging.debug(f'Could not inject metadata after {delay}s delay: %s', e)
		
		# Start delayed injection in background
		asyncio.create_task(delayed_injection())
		
		history = await agent.run(max_steps=args.max_steps)
		
		# Inject metadata again after agent has finished navigating and running
		# This ensures it's available even if the initial injection was lost
		if browser_instance:
			try:
				await browser_instance.execute_script(metadata_script)
			except Exception as e:
				if args.verbose:
					logging.debug('Could not re-inject metadata after agent run: %s', e)
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

	# Calculate and log cost information
	total_cost = 0.0
	average_cost_per_puzzle = 0.0
	
	if history.usage and history.usage.total_cost is not None:
		total_cost = float(history.usage.total_cost)
		logging.info('Total token cost: $%.6f', total_cost)
		
		# Try to get puzzle count from browser context
		browser_instance = browser if browser else (getattr(agent, 'browser', None) if hasattr(agent, 'browser') else None)
		if browser_instance:
			try:
				puzzle_count_result = await browser_instance.evaluate("""
					(() => {
						if (window.benchmarkStats && window.benchmarkStats.total) {
							return window.benchmarkStats.total;
						}
						return 0;
					})()
				""")
				if puzzle_count_result and isinstance(puzzle_count_result, (int, float)):
					puzzle_count = int(puzzle_count_result)
			except Exception as e:
				if args.verbose:
					logging.debug('Could not get puzzle count from browser: %s', e)
		
		# Calculate average cost per puzzle
		if puzzle_count > 0:
			average_cost_per_puzzle = total_cost / puzzle_count
			logging.info('Average cost per puzzle: $%.6f (based on %d puzzles)', average_cost_per_puzzle, puzzle_count)
		else:
			logging.warning('Could not determine puzzle count, cannot calculate average cost per puzzle')
		
		# Inject final cost data and metadata into browser page for JavaScript to use
		if browser_instance:
			try:
				final_data_script = f"""
				window.__agentCostData = {{
					totalCost: {total_cost},
					averageCostPerPuzzle: {average_cost_per_puzzle},
					puzzleCount: {puzzle_count}
				}};
				// Update cost tracker if it exists
				if (window.__agentCostTracker) {{
					window.__agentCostTracker.totalCost = {total_cost};
					window.__agentCostTracker.puzzleCount = {puzzle_count};
				}}
				// Ensure metadata is set
				if (!window.__agentMetadata) {{
					window.__agentMetadata = {{
						model: {json.dumps(model_name)},
						provider: {json.dumps(provider_name)},
						agentFramework: "browser-use"
					}};
				}}
				"""
				await browser_instance.execute_script(final_data_script)
			except Exception as e:
				if args.verbose:
					logging.debug('Could not inject final cost/metadata data: %s', e)

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
