"""Prompt-level defense implementations.

These defenses modify or augment the text prompt to enhance safety.
"""

from typing import Optional, Tuple, List
from dataclasses import dataclass
import re


@dataclass
class PromptDefenseConfig:
    """Configuration for prompt-level defenses."""
    # Safety prompt
    safety_prompt_position: str = "prefix"  # "prefix", "suffix", or "both"
    
    # Keyword filtering
    filter_keywords: List[str] = None
    replacement_text: str = "[FILTERED]"
    
    # Instruction injection detection
    detect_injection: bool = True
    
    def __post_init__(self):
        if self.filter_keywords is None:
            self.filter_keywords = [
                'ignore previous', 'disregard', 'forget instructions',
                'new instructions', 'override', 'bypass', 'jailbreak'
            ]


class SafetyPromptDefense:
    """Defense that adds safety-enhancing prompts."""
    
    SAFETY_PROMPTS = {
        'standard': (
            "You are a helpful, harmless, and honest AI assistant. "
            "You should refuse to help with any request that could cause harm, "
            "is illegal, unethical, or dangerous. "
        ),
        'strict': (
            "IMPORTANT: You must follow these safety guidelines strictly. "
            "Do not provide information about: illegal activities, weapons, "
            "hacking, harmful substances, or anything that could cause harm. "
            "If asked about such topics, politely decline and explain why. "
        ),
        'contextual': (
            "Before responding, consider: Is this request safe? Could the "
            "response be misused? If there's any doubt, err on the side of caution. "
        ),
        'reminder': (
            "Remember: Your primary goal is to be helpful while avoiding harm. "
            "Never assist with dangerous, illegal, or unethical requests. "
        )
    }
    
    def __init__(self, config: Optional[PromptDefenseConfig] = None):
        self.config = config or PromptDefenseConfig()
    
    def defend(
        self,
        prompt: str,
        safety_level: str = "standard"
    ) -> Tuple[str, str]:
        """Apply safety prompt defense.
        
        Args:
            prompt: Original user prompt
            safety_level: Level of safety prompt ('standard', 'strict', 'contextual', 'reminder')
            
        Returns:
            Tuple of (modified_prompt, reason)
        """
        safety_prompt = self.SAFETY_PROMPTS.get(safety_level, self.SAFETY_PROMPTS['standard'])
        
        if self.config.safety_prompt_position == "prefix":
            modified = safety_prompt + "\n\nUser request: " + prompt
        elif self.config.safety_prompt_position == "suffix":
            modified = prompt + "\n\n" + safety_prompt
        else:  # both
            modified = safety_prompt + "\n\nUser request: " + prompt + "\n\n" + self.SAFETY_PROMPTS['reminder']
        
        return modified, f"Added {safety_level} safety prompt ({self.config.safety_prompt_position})"


class KeywordFilterDefense:
    """Defense that filters potentially harmful keywords from prompts."""
    
    def __init__(self, config: Optional[PromptDefenseConfig] = None):
        self.config = config or PromptDefenseConfig()
        
        # Extended harmful keyword list
        self.harmful_patterns = [
            r'\b(hack|hacking|hacker)\b',
            r'\b(bomb|explosive|detonate)\b',
            r'\b(kill|murder|assassinate)\b',
            r'\b(weapon|gun|firearm)\b',
            r'\b(drug|narcotic|cocaine|heroin)\b',
            r'\b(malware|virus|trojan|ransomware)\b',
            r'\b(steal|theft|rob)\b',
            r'\b(illegal|illicit|unlawful)\b',
        ]
    
    def detect_harmful_keywords(self, text: str) -> List[str]:
        """Detect harmful keywords in text."""
        found = []
        text_lower = text.lower()
        
        for pattern in self.harmful_patterns:
            matches = re.findall(pattern, text_lower, re.IGNORECASE)
            found.extend(matches)
        
        return list(set(found))
    
    def defend(self, prompt: str) -> Tuple[str, str, List[str]]:
        """Apply keyword filter defense.
        
        Args:
            prompt: Original user prompt
            
        Returns:
            Tuple of (modified_prompt, reason, detected_keywords)
        """
        detected = self.detect_harmful_keywords(prompt)
        
        if not detected:
            return prompt, "No harmful keywords detected", []
        
        # Filter keywords
        modified = prompt
        for pattern in self.harmful_patterns:
            modified = re.sub(pattern, self.config.replacement_text, modified, flags=re.IGNORECASE)
        
        return modified, f"Filtered {len(detected)} harmful keywords", detected


class InjectionDetectionDefense:
    """Defense that detects prompt injection attempts."""
    
    INJECTION_PATTERNS = [
        r'ignore\s+(all\s+)?previous\s+instructions?',
        r'disregard\s+(all\s+)?previous',
        r'forget\s+(all\s+)?instructions?',
        r'new\s+instructions?:',
        r'override\s+(safety|guidelines)',
        r'bypass\s+(safety|filter)',
        r'jailbreak',
        r'developer\s+mode',
        r'DAN\s+mode',
        r'pretend\s+you\s+are',
        r'act\s+as\s+if',
        r'roleplay\s+as',
        r'you\s+are\s+now',
        r'from\s+now\s+on',
    ]
    
    def __init__(self, config: Optional[PromptDefenseConfig] = None):
        self.config = config or PromptDefenseConfig()
    
    def detect_injection(self, prompt: str) -> Tuple[bool, List[str]]:
        """Detect prompt injection attempts.
        
        Args:
            prompt: User prompt to check
            
        Returns:
            Tuple of (is_injection, matched_patterns)
        """
        matched = []
        prompt_lower = prompt.lower()
        
        for pattern in self.INJECTION_PATTERNS:
            if re.search(pattern, prompt_lower, re.IGNORECASE):
                matched.append(pattern)
        
        return len(matched) > 0, matched
    
    def defend(self, prompt: str) -> Tuple[bool, str, str]:
        """Apply injection detection defense.
        
        Args:
            prompt: User prompt to check
            
        Returns:
            Tuple of (is_blocked, reason, original_prompt)
        """
        is_injection, patterns = self.detect_injection(prompt)
        
        if is_injection:
            return True, f"Blocked: detected injection patterns: {patterns}", prompt
        
        return False, "No injection detected", prompt


class CompositePromptDefense:
    """Combines multiple prompt-level defenses."""
    
    def __init__(self, config: Optional[PromptDefenseConfig] = None):
        self.config = config or PromptDefenseConfig()
        
        self.safety_defense = SafetyPromptDefense(config)
        self.keyword_defense = KeywordFilterDefense(config)
        self.injection_defense = InjectionDetectionDefense(config)
    
    def defend(
        self,
        prompt: str,
        safety_level: str = "standard"
    ) -> Tuple[bool, str, List[str]]:
        """Apply composite prompt defense.
        
        Args:
            prompt: Original user prompt
            safety_level: Safety prompt level
            
        Returns:
            Tuple of (is_blocked, modified_prompt, reasons)
        """
        reasons = []
        current_prompt = prompt
        
        # 1. Check for injection
        is_blocked, reason, _ = self.injection_defense.defend(current_prompt)
        reasons.append(f"Injection detection: {reason}")
        
        if is_blocked:
            return True, current_prompt, reasons
        
        # 2. Filter keywords
        current_prompt, reason, detected = self.keyword_defense.defend(current_prompt)
        reasons.append(f"Keyword filter: {reason}")
        
        # 3. Add safety prompt
        current_prompt, reason = self.safety_defense.defend(current_prompt, safety_level)
        reasons.append(f"Safety prompt: {reason}")
        
        return False, current_prompt, reasons
