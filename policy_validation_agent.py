from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from typing import List, Dict, Optional
import json
import logging
import os

class PolicyValidationAgent:
    def __init__(self, groq_api_key: str, model: str = "mixtral-8x7b-32768"):
        self.llm = ChatGroq(
            groq_api_key=groq_api_key,
            model_name=model,
            temperature=0.2,  # Lower temperature for more consistent policy validation
            max_tokens=2048
        )
        self.policies_data = self._load_policies()
        
    def _load_policies(self) -> Dict:
        """Load policy documents from text files"""
        policies = {}
        try:
            policies_dir = "policies_data"
            if not os.path.exists(policies_dir):
                os.makedirs(policies_dir)
                
            if os.path.exists(f"{policies_dir}/policies.txt"):
                with open(f"{policies_dir}/policies.txt", 'r') as f:
                    policies = json.loads(f.read())
                logging.info(f"Loaded {len(policies)} policies")
            else:
                logging.warning("No policies file found")
        except Exception as e:
            logging.error(f"Error loading policies: {str(e)}")
        return policies
    
    def _get_response(self, messages: List[dict]) -> str:
        try:
            response = self.llm.invoke(messages)
            return response.content
        except Exception as e:
            return f"Error getting response: {str(e)}"
    
    def validate_policy(self, policy_number: str, claim_details: Dict) -> Dict:
        """Validate if a claim is covered under the policy"""
        policy_data = self.policies_data.get(policy_number)
        
        if not policy_data:
            return {
                "valid": False,
                "error": "Policy number not found",
                "details": None
            }
        
        messages = [
            SystemMessage(content=f"""You are a Policy Validation Oracle specialized in insurance policy verification.
            
            Policy Details:
            {json.dumps(policy_data, indent=2)}
            
            Claim Details:
            {json.dumps(claim_details, indent=2)}
            
            Analyze the claim against the policy terms and provide a structured response including:
            1. Coverage Verification
            2. Policy Limits Check
            3. Exclusions Analysis
            4. Terms Compliance
            5. Documentation Requirements
            
            Format your response in clear sections with bullet points."""),
            HumanMessage(content="Validate this claim against the policy terms.")
        ]
        
        validation_result = self._get_response(messages)
        
        return {
            "valid": True,
            "policy_data": policy_data,
            "validation_details": validation_result
        }
    
    def check_policy_limits(self, policy_number: str, claim_amount: float) -> Dict:
        """Check if claim amount is within policy limits"""
        policy_data = self.policies_data.get(policy_number)
        
        if not policy_data:
            return {"valid": False, "error": "Policy not found"}
            
        coverage_limit = policy_data.get('coverage_limit', 0)
        remaining_coverage = policy_data.get('remaining_coverage', coverage_limit)
        
        return {
            "valid": claim_amount <= remaining_coverage,
            "coverage_limit": coverage_limit,
            "remaining_coverage": remaining_coverage,
            "claim_amount": claim_amount,
            "within_limit": claim_amount <= remaining_coverage
        }
    
    def verify_documentation(self, policy_number: str, claim_type: str, provided_docs: List[str]) -> Dict:
        """Verify if all required documents are provided"""
        policy_data = self.policies_data.get(policy_number)
        
        if not policy_data:
            return {"valid": False, "error": "Policy not found"}
            
        messages = [
            SystemMessage(content=f"""Verify documentation requirements for this claim.
            
            Policy Requirements:
            {json.dumps(policy_data.get('documentation_requirements', {}), indent=2)}
            
            Claim Type: {claim_type}
            Provided Documents: {json.dumps(provided_docs, indent=2)}
            
            Check:
            1. Required vs Provided Documents
            2. Missing Documents
            3. Additional Requirements"""),
            HumanMessage(content="Verify documentation completeness.")
        ]
        
        verification_result = self._get_response(messages)
        
        return {
            "verification_result": verification_result,
            "provided_documents": provided_docs,
            "policy_requirements": policy_data.get('documentation_requirements', {})
        }
    
    def get_policy_summary(self, policy_number: str) -> str:
        """Get a human-readable summary of policy terms"""
        policy_data = self.policies_data.get(policy_number)
        
        if not policy_data:
            return "Policy not found"
            
        messages = [
            SystemMessage(content=f"""Create a clear, concise summary of this insurance policy.
            
            Policy Details:
            {json.dumps(policy_data, indent=2)}
            
            Include:
            1. Coverage Overview
            2. Key Terms and Conditions
            3. Important Exclusions
            4. Claim Requirements
            5. Coverage Limits"""),
            HumanMessage(content="Generate a policy summary.")
        ]
        
        return self._get_response(messages)