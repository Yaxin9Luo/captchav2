"""
LLM Response Logger for capturing detailed LLM outputs during agent runs.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any


class LLMResponseLogger:
	"""Logger to capture and save detailed LLM responses."""

	def __init__(self, log_dir: str = 'llm_logs', experiment_name: str = None):
		self.log_dir = Path(log_dir)
		self.log_dir.mkdir(exist_ok=True)

		timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
		exp_name = f'_{experiment_name}' if experiment_name else ''
		self.log_file = self.log_dir / f'llm_responses_{timestamp}{exp_name}.jsonl'
		self.step_counter = 0

		logging.info(f'LLM responses will be logged to: {self.log_file}')

	def log_response(self, step: int, response: Any, metadata: dict = None):
		"""Log a single LLM response with full details."""
		self.step_counter += 1
		log_entry = {
			'step': step,
			'counter': self.step_counter,
			'timestamp': datetime.now().isoformat(),
			'metadata': metadata or {},
		}

		# Try to extract detailed information from the response
		# Handle different response types
		if hasattr(response, 'content'):
			log_entry['content'] = str(response.content)
		if hasattr(response, 'text'):
			log_entry['text'] = str(response.text)
		if hasattr(response, 'message'):
			log_entry['message'] = str(response.message)
		if isinstance(response, str):
			log_entry['text'] = response

		# Extract usage info
		if hasattr(response, 'usage'):
			try:
				log_entry['usage'] = {
					'prompt_tokens': getattr(response.usage, 'prompt_tokens', None),
					'completion_tokens': getattr(response.usage, 'completion_tokens', None),
					'total_tokens': getattr(response.usage, 'total_tokens', None),
				}
			except Exception:
				log_entry['usage'] = str(response.usage)

		# Extract model info
		if hasattr(response, 'model'):
			log_entry['model'] = str(response.model)

		# Extract tool calls
		if hasattr(response, 'tool_calls'):
			try:
				log_entry['tool_calls'] = [
					{
						'id': getattr(tc, 'id', None),
						'type': getattr(tc, 'type', None),
						'function': {
							'name': getattr(tc.function, 'name', None) if hasattr(tc, 'function') else None,
							'arguments': getattr(tc.function, 'arguments', None) if hasattr(tc, 'function') else None,
						}
					}
					for tc in response.tool_calls
				] if response.tool_calls else []
			except Exception:
				log_entry['tool_calls'] = str(response.tool_calls)

		# Capture raw response for debugging
		if not any(k in log_entry for k in ['content', 'text', 'message']):
			log_entry['raw_response'] = str(response)
			# Try to get dict representation
			if hasattr(response, '__dict__'):
				try:
					log_entry['response_dict'] = {
						k: str(v) for k, v in response.__dict__.items()
					}
				except Exception:
					pass

		# Write to file
		with open(self.log_file, 'a', encoding='utf-8') as f:
			f.write(json.dumps(log_entry, ensure_ascii=False, indent=None) + '\n')

		# Also log summary to console
		logging.info(f'[LLM Response #{self.step_counter} @ Step {step}] Logged to file')

		return log_entry

	def log_history_item(self, item: Any, index: int = None):
		"""Log a history item from the agent's conversation history."""
		log_entry = {
			'type': 'history_item',
			'index': index,
			'counter': self.step_counter + 1,
			'timestamp': datetime.now().isoformat(),
		}

		# Extract role and content
		if hasattr(item, 'role'):
			log_entry['role'] = str(item.role)
		if hasattr(item, 'content'):
			log_entry['content'] = str(item.content)
		if hasattr(item, 'text'):
			log_entry['text'] = str(item.text)

		# Try to get full dict
		if hasattr(item, '__dict__'):
			try:
				log_entry['item_dict'] = {
					k: str(v) for k, v in item.__dict__.items()
				}
			except Exception:
				log_entry['raw_item'] = str(item)
		else:
			log_entry['raw_item'] = str(item)

		# Write to file
		with open(self.log_file, 'a', encoding='utf-8') as f:
			f.write(json.dumps(log_entry, ensure_ascii=False, indent=None) + '\n')

	def close(self):
		"""Close the logger and write summary."""
		summary = {
			'type': 'summary',
			'total_responses': self.step_counter,
			'timestamp': datetime.now().isoformat(),
			'log_file': str(self.log_file),
		}

		with open(self.log_file, 'a', encoding='utf-8') as f:
			f.write(json.dumps(summary, ensure_ascii=False) + '\n')

		logging.info(f'LLM logger closed. Total responses logged: {self.step_counter}')


class LoggingLLMWrapper:
	"""Wrapper that logs LLM calls in real-time while forwarding to the actual LLM."""

	def __init__(self, llm: Any, logger: LLMResponseLogger):
		self.llm = llm
		self.logger = logger
		self.call_counter = 0

		# Copy over all attributes from the wrapped LLM
		for attr in dir(llm):
			if not attr.startswith('_') and not hasattr(self, attr):
				try:
					setattr(self, attr, getattr(llm, attr))
				except AttributeError:
					pass

	def __getattr__(self, name: str) -> Any:
		"""Forward attribute access to the wrapped LLM."""
		return getattr(self.llm, name)

	async def ainvoke(self, messages: Any, output_format: Any = None, **kwargs: Any) -> Any:
		"""Intercept async LLM calls and log them in real-time."""
		self.call_counter += 1
		call_id = self.call_counter

		# Log the input messages
		try:
			# Try to extract more structured info from messages
			messages_data = []
			if isinstance(messages, list):
				for msg in messages:
					if hasattr(msg, 'content'):
						messages_data.append({'role': getattr(msg, 'role', 'unknown'), 'content': str(msg.content)[:500]})
					else:
						messages_data.append(str(msg)[:500])
			else:
				messages_data = str(messages)[:1000]

			self.logger.log_response(
				step=call_id,
				response={'type': 'request', 'messages': messages_data, 'kwargs': str(kwargs)[:200]},
				metadata={'call_id': call_id, 'direction': 'request'}
			)
		except Exception as e:
			logging.warning(f'Failed to log LLM request: {e}')

		# Call the actual LLM
		try:
			response = await self.llm.ainvoke(messages, output_format, **kwargs)

			# Log the response immediately after receiving it
			try:
				self.logger.log_response(
					step=call_id,
					response=response,
					metadata={'call_id': call_id, 'direction': 'response'}
				)
			except Exception as e:
				logging.warning(f'Failed to log LLM response: {e}')

			return response

		except Exception as e:
			# Log the error
			try:
				self.logger.log_response(
					step=call_id,
					response={'type': 'error', 'error': str(e)},
					metadata={'call_id': call_id, 'direction': 'error'}
				)
			except Exception as log_err:
				logging.warning(f'Failed to log LLM error: {log_err}')

			# Re-raise the original error
			raise

	def invoke(self, messages: Any, output_format: Any = None, **kwargs: Any) -> Any:
		"""Intercept sync LLM calls and log them in real-time."""
		self.call_counter += 1
		call_id = self.call_counter

		# Log the input messages
		try:
			# Try to extract more structured info from messages
			messages_data = []
			if isinstance(messages, list):
				for msg in messages:
					if hasattr(msg, 'content'):
						messages_data.append({'role': getattr(msg, 'role', 'unknown'), 'content': str(msg.content)[:500]})
					else:
						messages_data.append(str(msg)[:500])
			else:
				messages_data = str(messages)[:1000]

			self.logger.log_response(
				step=call_id,
				response={'type': 'request', 'messages': messages_data, 'kwargs': str(kwargs)[:200]},
				metadata={'call_id': call_id, 'direction': 'request'}
			)
		except Exception as e:
			logging.warning(f'Failed to log LLM request: {e}')

		# Call the actual LLM
		try:
			response = self.llm.invoke(messages, output_format, **kwargs)

			# Log the response immediately after receiving it
			try:
				self.logger.log_response(
					step=call_id,
					response=response,
					metadata={'call_id': call_id, 'direction': 'response'}
				)
			except Exception as e:
				logging.warning(f'Failed to log LLM response: {e}')

			return response

		except Exception as e:
			# Log the error
			try:
				self.logger.log_response(
					step=call_id,
					response={'type': 'error', 'error': str(e)},
					metadata={'call_id': call_id, 'direction': 'error'}
				)
			except Exception as log_err:
				logging.warning(f'Failed to log LLM error: {log_err}')

			# Re-raise the original error
			raise
