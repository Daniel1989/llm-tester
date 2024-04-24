import json
import re
import subprocess
import uuid
from os.path import sep

from exceptions import TokenLimitError, ApiError
from function_calling import parse_agent_response, FunctionCallSet
from llm import create_gpt_chat_completion
from utils import get_prompt, get_sys_message, capitalize_first_word_with_underscores
from const.llm import END_RESPONSE
from cli import running_processes


class AgentConvo:
    """
    Represents a conversation with an agent.

    Args:
        agent: An instance of the agent participating in the conversation.
    """

    def __init__(self, agent, temperature: float = 0.7):
        # [{'role': 'system'|'user'|'assistant', 'content': ''}, ...]
        self.messages: list[dict] = []
        self.branches = {}
        self.log_to_user = True
        self.agent = agent
        self.high_level_step = 'start'
        self.temperature = temperature

        # add system message
        system_message = get_sys_message(self.agent.role)
        self.messages.append(system_message)

    def send_message(self, prompt_path=None, prompt_data=None, function_calls: FunctionCallSet = None, should_log_message=True):
        """
        Sends a message in the conversation.

        Args:
            prompt_path: The path to a prompt.
            prompt_data: Data associated with the prompt.
            function_calls: Optional function calls to be included in the message.
            should_log_message: Flag if final response should be logged.
        Returns:
            The response from the agent.
        """
        # craft message
        self.construct_and_add_message_from_prompt(prompt_path, prompt_data)

        # TODO: move this if block (and the other below) to Developer agent - https://github.com/Pythagora-io/gpt-pilot/issues/91#issuecomment-1751964079
        # check if we already have the LLM response saved
        if hasattr(self.agent, 'save_dev_steps') and self.agent.save_dev_steps:
            self.agent.project.llm_req_num += 1

        try:
            response = create_gpt_chat_completion(self.messages, self.high_level_step,
                                                  function_calls=function_calls, prompt_data=prompt_data,
                                                  temperature=self.temperature)
        except TokenLimitError as e:
            raise e

        # TODO handle errors from OpenAI
        # It's complicated because calling functions are expecting different types of responses - string or tuple
        # https://github.com/Pythagora-io/gpt-pilot/issues/165 & #91
        if response == {} or response is None:
            # This should never happen since we're raising ApiError in create_gpt_chat_completion
            # Leaving this in place in case there's a case where this can still happen
            payload_size_kb = len(json.dumps(self.messages)) // 1000
            raise ApiError(f"Unknown API error (prompt: {prompt_path}, request size: {payload_size_kb}KB)")

        try:
            response = parse_agent_response(response, function_calls)
        except (KeyError, json.JSONDecodeError) as err:
            raise ApiError(f"Error parsing LLM response: {err.__class__.__name__}: {err}: Response text: {response}") from err

        message_content = self.format_message_content(response, function_calls)

        self.messages.append({"role": "assistant", "content": message_content})

        return response

    def format_message_content(self, response, function_calls):
        # TODO remove this once the database is set up properly
        if isinstance(response, str):
            return response
        else:
            # string_response = []
            # for key, value in response.items():
            #     string_response.append(f'# {key}')
            #
            #     if isinstance(value, list):
            #         if 'to_message' in function_calls:
            #             string_response.append(function_calls['to_message'](value))
            #         elif len(value) > 0 and isinstance(value[0], dict):
            #             string_response.extend([
            #                 f'##{i}\n' + array_of_objects_to_string(d)
            #                 for i, d in enumerate(value)
            #             ])
            #         else:
            #             string_response.extend(['- ' + r for r in value])
            #     else:
            #         string_response.append(str(value))
            #
            # return '\n'.join(string_response)
            return json.dumps(response)
        # TODO END

    def continuous_conversation(self, prompt_path, prompt_data, function_calls=None):
        """
        Conducts a continuous conversation with the agent.

        Args:
            prompt_path: The path to a prompt.
            prompt_data: Data associated with the prompt.
            function_calls: Optional function calls to be included in the conversation.

        Returns:
            List of accepted messages in the conversation.
        """
        self.log_to_user = False
        accepted_messages = []
        response = self.send_message(prompt_path, prompt_data, function_calls)

        # Continue conversation until GPT response equals END_RESPONSE
        while response != END_RESPONSE:
            print("Do you want to add anything else? If not, just press ENTER.")
            user_message = ""

            if user_message == "":
                accepted_messages.append(response)

            self.messages.append({"role": "user", "content": user_message})
            response = self.send_message(None, None, function_calls)

        self.log_to_user = True
        return accepted_messages

    def save_branch(self, branch_name=None):
        if branch_name is None:
            branch_name = str(uuid.uuid4())
        self.branches[branch_name] = self.messages.copy()
        return branch_name

    def load_branch(self, branch_name, reload_files=True):
        self.messages = self.branches[branch_name].copy()
        if reload_files:
            # TODO make this more flexible - with every message, save metadata so every time we load a branch, reconstruct all messages from scratch
            self.replace_files()

    def replace_files(self):
        relevant_files = getattr(self.agent, 'relevant_files', None)
        files = self.agent.project.get_all_coded_files(relevant_files=relevant_files)
        for msg in self.messages:
            if msg['role'] == 'user':
                new_content = self.replace_files_in_one_message(files, msg["content"])
                if new_content != msg["content"]:
                    msg["content"] = new_content

    def replace_files_in_one_message(self, files, message):
        # This needs to EXACTLY match the formatting in `files_list.prompt`
        replacement_lines = ["\n---START_OF_FILES---"]
        for file in files:
            path = f"{file['path']}{sep}{file['name']}"
            content = file['content']
            replacement_lines.append(f"**{path}** ({ file['lines_of_code'] } lines of code):\n```\n{content}\n```\n")
        replacement_lines.append("---END_OF_FILES---\n")
        replacement = "\n".join(replacement_lines)

        def replace_cb(_m):
            return replacement

        pattern = r"\n---START_OF_FILES---\n(.*?)\n---END_OF_FILES---\n"
        return re.sub(pattern, replace_cb, message, flags=re.MULTILINE|re.DOTALL)

    @staticmethod
    def escape_specials(s):
        s = s.replace("\\", "\\\\")

        # List of sequences to preserve
        sequences_to_preserve = [
            # todo check if needed "\\\\",  # Backslash - note: probably not eg. paths on Windows
            "\\'",  # Single quote
            '\\"',  # Double quote
            # todo check if needed '\\a',  # ASCII Bell (BEL)
            # todo check if needed '\\b',  # ASCII Backspace (BS) - note: different from regex \b
            # todo check if needed '\\f',  # ASCII Formfeed (FF)
            '\\n',  # ASCII Linefeed (LF)
            # todo check if needed '\\r',  # ASCII Carriage Return (CR)
            '\\t',  # ASCII Horizontal Tab (TAB)
            # todo check if needed '\\v'  # ASCII Vertical Tab (VT)
        ]

        for seq in sequences_to_preserve:
            s = s.replace('\\\\' + seq[-1], seq)
        return s

    def convo_length(self):
        return len([msg for msg in self.messages if msg['role'] != 'system'])

    def log_message(self, content):
        """
        Logs a message in the conversation.

        Args:
            content: The content of the message to be logged.
        """
        print_msg = capitalize_first_word_with_underscores(self.high_level_step)
        if self.log_to_user:
            if self.agent.project.checkpoints['last_development_step'] is not None:
                dev_step_msg = f'\nDev step {str(self.agent.project.checkpoints["last_development_step"]["id"])}\n'
                print(dev_step_msg)
            try:
                print(f"\n{content}\n")
            except Exception:  # noqa
                # Workaround for Windows encoding crash: https://github.com/Pythagora-io/gpt-pilot/issues/509
                safe_content = content.encode('ascii', 'ignore').decode('ascii')
                print(f"\n{safe_content}\n")


    def to_context_prompt(self):

        # TODO: get dependencies & versions from the project (package.json, requirements.txt, pom.xml, etc.)
        # Ideally, the LLM could do this, and we update it on load & whenever the file changes
        # ...or LLM generates a script for `.gpt-pilot/get_dependencies` that we run
        # https://github.com/Pythagora-io/gpt-pilot/issues/189
        return get_prompt('development/context.prompt', {
            'directory_tree': self.agent.project.get_directory_tree(),
            'running_processes': running_processes,
        })

    def to_playground(self):
        # Internal function to help debugging in OpenAI Playground, not to be used in production
        with open('const/convert_to_playground_convo.js', 'r', encoding='utf-8') as file:
            content = file.read()
        process = subprocess.Popen('pbcopy', stdin=subprocess.PIPE)
        process.communicate(content.replace('{{messages}}', str(self.messages)).encode('utf-8'))

    def remove_last_x_messages(self, x):
        self.messages = self.messages[:-x]

    def construct_and_add_message_from_prompt(self, prompt_path, prompt_data):
        if prompt_path is not None and prompt_data is not None:
            prompt = get_prompt(prompt_path, prompt_data)
            self.messages.append({"role": "user", "content": prompt})
