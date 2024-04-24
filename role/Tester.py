
from const.function_calls import ARCHITECTURE
import platform
from agent.AgentConvo import AgentConvo

ARCHITECTURE_STEP = 'architecture'
WARN_SYSTEM_DEPS = ["docker", "kubernetes", "microservices"]
WARN_FRAMEWORKS = ["next.js", "vue", "vue.js", "svelte", "angular"]
WARN_FRAMEWORKS_URL = "https://github.com/Pythagora-io/gpt-pilot/wiki/Using-GPT-Pilot-with-frontend-frameworks"


class Tester():
    def __init__(self):
        super().__init__()
        self.role = 'architect'
        self.convo_architecture = None

    def getting_started(self):
        self.convo_architecture = AgentConvo(self)
        llm_response = self.convo_architecture.send_message('architecture/technologies.prompt',
                                                            {"os": platform.system(),
                                                             },
                                                            ARCHITECTURE
                                                            )

        print("gpt response is\n", llm_response)
