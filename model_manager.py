
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
try:
    from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
except ImportError:
    ChatHuggingFace = None
    HuggingFaceEndpoint = None

load_dotenv()

class ModelManager:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelManager, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        self.provider = os.getenv("MODEL_PROVIDER", "groq").lower()
        self.google_api_key = os.getenv("GOOGLE_API_KEY")
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        
        # Default Model IDs
        self.fast_model_id = os.getenv("FAST_MODEL", "openai/gpt-oss-20b")
        self.smart_model_id = os.getenv("SMART_MODEL", "openai/gpt-oss-120b")
        self.vision_model_id = os.getenv("VISION_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct")
        
        # Initialize instances
        self.fast_llm = None
        self.smart_llm = None
        self.vision_llm = None
        
        self._load_models()

    def _load_models(self):
        print(f"🔄 ModelManager: Loading models (Provider: {self.provider})...")
        
        if self.provider == "huggingface":
            if not ChatHuggingFace:
                raise ImportError("langchain-huggingface required for huggingface provider")
            
            self.fast_llm = ChatHuggingFace(llm=HuggingFaceEndpoint(
                repo_id=self.fast_model_id,
                task="text-generation",
                max_new_tokens=512,
                do_sample=False,
            ))
            
            self.smart_llm = ChatHuggingFace(llm=HuggingFaceEndpoint(
                repo_id=self.smart_model_id,
                task="text-generation",
                max_new_tokens=1024,
                do_sample=True,
                temperature=0.5
            ))
            # Vision fallback
            if self.google_api_key:
                self.vision_llm = ChatGoogleGenerativeAI(
                    model="gemini-1.5-flash",
                    google_api_key=self.google_api_key,
                    temperature=0.2
                )
                
        elif self.provider == "groq":
            self.fast_llm = ChatGroq(
                model_name=self.fast_model_id,
                groq_api_key=self.groq_api_key,
                temperature=0,
                max_retries=2
            )
            
            self.smart_llm = ChatGroq(
                model_name=self.smart_model_id,
                groq_api_key=self.groq_api_key,
                temperature=0.5,
                max_retries=3,
                reasoning_effort="medium"
            )
            
            self.vision_llm = ChatGroq(
                model_name=self.vision_model_id,
                groq_api_key=self.groq_api_key,
                temperature=0.2
            )
            
        else: # Google
            self.fast_llm = ChatGoogleGenerativeAI(
                model=self.fast_model_id, # "gemini-1.5-flash"
                google_api_key=self.google_api_key,
                temperature=0,
                max_retries=2,
                convert_system_message_to_human=True
            )
            
            self.smart_llm = ChatGoogleGenerativeAI(
                model=self.smart_model_id, # "gemini-1.5-pro"
                google_api_key=self.google_api_key,
                temperature=0.5,
                max_retries=3
            )
            
            self.vision_llm = ChatGoogleGenerativeAI(
                model="gemini-1.5-flash",
                google_api_key=self.google_api_key,
                temperature=0.2
            )
            
        print(f"✅ Models Loaded: Fast={self.fast_model_id}, Smart={self.smart_model_id}")

    def get_model(self, model_type: str = "fast"):
        """
        Get the LLM instance for the specified type.
        Args:
            model_type: "fast", "smart", or "vision"
        """
        if model_type == "smart":
            return self.smart_llm
        elif model_type == "vision":
            return self.vision_llm
        else:
            return self.fast_llm

    def update_model(self, model_type: str, new_model_id: str, persist: bool = False):
        """
        Dynamically update the model ID for a specific type and reload.
        If persist is True, saves the change to .env.
        """
        print(f"🔄 Updating {model_type} model to {new_model_id}...")
        if model_type == "fast":
            self.fast_model_id = new_model_id
            env_key = "FAST_MODEL"
        elif model_type == "smart":
            self.smart_model_id = new_model_id
            env_key = "SMART_MODEL"
        elif model_type == "vision":
            self.vision_model_id = new_model_id
            env_key = "VISION_MODEL"
        else:
            return False
            
        self._load_models()
        
        if persist:
            self._save_to_env(env_key, new_model_id)
            
        return True

    def _save_to_env(self, key: str, value: str):
        """Helper to update .env file."""
        try:
            env_path = ".env"
            lines = []
            if os.path.exists(env_path):
                with open(env_path, "r", encoding="utf-8") as f:
                    lines = f.readlines()
            
            key_found = False
            new_lines = []
            for line in lines:
                if line.strip().startswith(f"{key}="):
                    new_lines.append(f"{key}={value}\n")
                    key_found = True
                else:
                    new_lines.append(line)
            
            if not key_found:
                if new_lines and not new_lines[-1].endswith("\n"):
                    new_lines.append("\n")
                new_lines.append(f"{key}={value}\n")
                
            with open(env_path, "w", encoding="utf-8") as f:
                f.writelines(new_lines)
            print(f"💾 Saved {key}={value} to .env")
        except Exception as e:
            print(f"Error saving to .env: {e}")

    def get_current_config(self):
        return {
            "provider": self.provider,
            "fast_model": self.fast_model_id,
            "smart_model": self.smart_model_id,
            "vision_model": self.vision_model_id
        }

# Global instance
model_manager = ModelManager()
