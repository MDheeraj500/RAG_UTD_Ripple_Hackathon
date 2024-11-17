from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from typing import List, Dict, Optional
import json
import logging
from claims_data_loader import ClaimsDataLoader
logging.basicConfig(level=logging.WARNING)

class ClaimsAnalysisAgent:
    def __init__(self, groq_api_key: str, model: str = "mixtral-8x7b-32768"):
        self.llm = ChatGroq(
            groq_api_key=groq_api_key,
            model_name=model,
            temperature=0.3,
            max_tokens=2048
        )
        self.policy_context = None
        self.data_loader = ClaimsDataLoader()
        self.claims_history = self.data_loader.load_claims_history()
        
    def set_policy_context(self, policy_data: Dict):
        self.policy_context = policy_data
        
    def _get_response(self, messages: List[dict]) -> str:
        try:
            if self.policy_context:
                # Add policy context message
                context_message = SystemMessage(content=f"""You are an Insurance Claims Analysis Oracle, a specialized AI system for analyzing insurance claims.
                
                Current Policy Information:
                Policy Type: {self.policy_context.get('policy_type', 'N/A')}
                Coverage Limits: ${self.policy_context.get('coverage_limit', 0):,.2f}
                Deductible: ${self.policy_context.get('deductible', 0):,.2f}
                Policy Status: {self.policy_context.get('status', 'Unknown')}
                
                Historical Claims Statistics:
                {json.dumps(self.data_loader.get_claim_statistics(), indent=2)}
                
                Key Guidelines:
                1. Analyze claim validity against policy terms
                2. Check for coverage limits and exclusions
                3. Identify potential fraud indicators
                4. Consider claim history patterns
                5. Recommend optimal settlement approaches
                
                Shape your analysis to focus on:
                - Policy compliance
                - Risk assessment
                - Cost optimization
                - Settlement recommendations
                """)
                messages.insert(0, context_message)
            
            response = self.llm.invoke(messages)
            return response.content
        except Exception as e:
            return f"Error generating response: {str(e)}"
    
    def analyze_claim(self, claim_details: Dict) -> str:
        """Analyze a new insurance claim and provide recommendations"""
        messages = [
            SystemMessage(content="""You are the Insurance Claims Analysis Oracle, a specialized system for evaluating insurance claims.
            
            Provide your analysis in the following format:
            
            1. 📋 Claim Overview
               - Basic claim details
               - Key factors identified
            
            2. 🔍 Policy Compliance Analysis
               - Coverage verification
               - Terms and conditions check
               - Exclusions assessment
            
            3. ⚠️ Risk Indicators
               - Fraud risk assessment
               - Claim pattern analysis
               - Documentation completeness
            
            4. 💰 Settlement Recommendation
               - Recommended action (approve/deny/adjust)
               - Suggested settlement amount
               - Justification
            
            5. 📊 Next Steps
               - Required documentation
               - Suggested investigation steps
               - Timeline recommendations
            
            Keep your analysis thorough but concise, focusing on actionable insights."""),
            HumanMessage(content=f"""Please analyze the following claim:
            
            Claim Details:
            Claim ID: {claim_details.get('claim_id')}
            Type: {claim_details.get('claim_type')}
            Amount: ${claim_details.get('amount', 0):,.2f}
            Description: {claim_details.get('description')}
            Date Filed: {claim_details.get('date_filed')}
            
            Additional Information:
            {claim_details.get('additional_info', 'None provided')}""")
        ]
        
        return self._get_response(messages)
    
    def get_similar_claims(self, claim_details: Dict) -> str:
        """Find and analyze similar historical claims"""
        similar_claims = self.data_loader.filter_claims(
            claim_type=claim_details.get('claim_type'),
            min_amount=claim_details.get('amount') * 0.8,
            max_amount=claim_details.get('amount') * 1.2
        )
        
        messages = [
            SystemMessage(content="""Analyze the following similar historical claims and provide insights.
            Focus on:
            1. Settlement patterns
            2. Processing times
            3. Common documentation requirements
            4. Risk factors and red flags
            5. Optimization opportunities
            
            Format your response with clear sections and bullet points."""),
            HumanMessage(content=f"""
            Current Claim:
            {json.dumps(claim_details, indent=2)}
            
            Similar Historical Claims:
            {json.dumps(similar_claims, indent=2)}
            """)
        ]
        
        return self._get_response(messages)
    
    def save_claim_result(self, claim_details: Dict, analysis_result: str):
        """Save analyzed claim to history"""
        claim_record = {
            **claim_details,
            "analysis": analysis_result,
            "status": "Pending",
            "processing_time": 0,
            "documents_provided": []
        }
        self.data_loader.save_new_claim(claim_record)
    
    def validate_documentation(self, documents: List[Dict]) -> str:
        """Validate claim documentation completeness"""
        messages = [
            SystemMessage(content="""Review the provided documentation for completeness and validity.
            Check for:
            1. Required forms and evidence
            2. Documentation gaps
            3. Verification needs
            4. Additional documentation recommendations"""),
            HumanMessage(content=f"Validate the following documentation: {json.dumps(documents)}")
        ]
        
        return self._get_response(messages)
    
    def suggest_optimization(self, claim_details: Dict) -> str:
        """Suggest ways to optimize the claim settlement"""
        messages = [
            SystemMessage(content="""Analyze the claim for settlement optimization opportunities.
            Consider:
            1. Cost reduction opportunities
            2. Process efficiency improvements
            3. Risk mitigation strategies
            4. Settlement timing recommendations"""),
            HumanMessage(content=f"Suggest optimization strategies for: {json.dumps(claim_details)}")
        ]
        
        return self._get_response(messages)