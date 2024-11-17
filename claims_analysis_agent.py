from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from typing import List, Dict, Optional
import json
import logging
from datetime import datetime
from utils.data_loader import DataLoader

logging.basicConfig(level=logging.WARNING)

class ClaimsAnalysisAgent:
    def __init__(self, groq_api_key: str, model: str = "mixtral-8x7b-32768"):
        self.llm = ChatGroq(
            groq_api_key=groq_api_key,
            model_name=model,
            temperature=0.3,
            max_tokens=2048
        )
        self.data_loader = DataLoader()
        self.policy_context = None

    def set_policy_context(self, policy_data: Dict):
        """Set the insurance policy context for analysis"""
        self.policy_context = policy_data

    def _get_response(self, messages: List[dict]) -> str:
        try:
            if self.policy_context:
                context_message = SystemMessage(content=f"""You are the Claims Analysis Oracle, a specialized AI system for analyzing insurance claims.
                
                Current Policy Information:
                Policy Type: {self.policy_context.get('policy_type', 'N/A')}
                Coverage Limits: ${self.policy_context.get('coverage_limit', 0):,.2f}
                Deductible: ${self.policy_context.get('deductible', 0):,.2f}
                Policy Status: {self.policy_context.get('status', 'Unknown')}
                
                Key Guidelines:
                1. Analyze claim validity against policy terms
                2. Check for coverage limits and exclusions
                3. Identify potential fraud indicators
                4. Consider claim history patterns
                5. Recommend optimal settlement approaches
                """)
                messages.insert(0, context_message)
            
            response = self.llm.invoke(messages)
            return response.content
        except Exception as e:
            logging.error(f"Error in _get_response: {str(e)}")
            return f"Error generating response: {str(e)}"

    def analyze_claim(self, claim_details: Dict) -> str:
        """Analyze a new insurance claim and provide recommendations"""
        similar_claims = self.data_loader.search_claims(
            claim_type=claim_details.get('claim_type')
        )
        
        claim_history = self.data_loader.search_claims(
            policy_number=claim_details.get('policy_number')
        )
        
        messages = [
            SystemMessage(content=f"""Analyze this insurance claim thoroughly.
            
            Claim Details:
            {json.dumps(claim_details, indent=2)}
            
            Similar Claims History:
            {json.dumps(similar_claims, indent=2)}
            
            Policy Claim History:
            {json.dumps(claim_history, indent=2)}
            
            Provide a detailed analysis in the following format:
            
            1. 📋 Claim Overview
               - Basic details analysis
               - Completeness check
               - Initial risk assessment
            
            2. 🔍 Historical Analysis
               - Comparison with similar claims
               - Policy claim history patterns
               - Typical settlement ranges
            
            3. ⚠️ Risk Assessment
               - Fraud indicators (if any)
               - Documentation completeness
               - Unusual patterns
            
            4. 💰 Cost Analysis
               - Reasonableness of amount
               - Comparison with typical costs
               - Potential adjustments needed
            
            5. 📊 Settlement Recommendation
               - Clear approve/deny/adjust decision
               - Recommended settlement amount
               - Justification for recommendation
            
            6. 📝 Processing Notes
               - Special handling requirements
               - Additional documentation needed
               - Expected timeline
               - Follow-up actions"""),
            HumanMessage(content="Please analyze this claim and provide recommendations")
        ]
        
        return self._get_response(messages)

    def get_similar_claims(self, claim_details: Dict) -> str:
        """Find and analyze similar historical claims"""
        similar_claims = self.data_loader.search_claims(
            policy_number=claim_details.get('policy_number'),
            claim_type=claim_details.get('claim_type')
        )
        
        if not similar_claims:
            return f"""
            🔍 No similar claims found for:
            - Policy: {claim_details.get('policy_number')}
            - Type: {claim_details.get('claim_type')}
            
            This appears to be the first claim of this type for this policy.
            """
        
        messages = [
            SystemMessage(content=f"""Analyze these similar claims and provide insights.
            
            Similar Claims:
            {json.dumps(similar_claims, indent=2)}
            
            Provide analysis in the following format:
            1. 📊 Statistical Summary
            2. 🔍 Pattern Analysis
            3. 💡 Insights
            4. ⚠️ Risk Factors
            """),
            HumanMessage(content="Analyze similar claims patterns")
        ]
        
        analysis = self._get_response(messages)
        
        stats = self.data_loader.get_claim_statistics(claim_details.get('policy_number'))
        
        summary_stats = f"""
        📊 Quick Statistics:
        • Total Claims: {stats.get('total_claims', 0)}
        • Approval Rate: {stats.get('approval_rate', 0):.1f}%
        • Average Amount: ${stats.get('average_amount', 0):,.2f}
        • Average Processing Time: {stats.get('average_processing_time', 0):.1f} days
        
        {analysis}
        """
        
        return summary_stats

    def detect_fraud_indicators(self, claim_details: Dict) -> List[str]:
        """Detect potential fraud indicators in a claim"""
        red_flags = []
        
        # Get recent claims for this policy
        recent_claims = self.data_loader.search_claims(
            policy_number=claim_details.get('policy_number'),
            date_from=(datetime.strptime(claim_details.get('date_filed'), '%Y-%m-%d')
                      .replace(month=1).strftime('%Y-%m-%d'))
        )
        
        # Check for multiple claims in short period
        if len(recent_claims) > 5:
            red_flags.append("High frequency of claims")
        
        # Get similar claims for amount comparison
        similar_claims = self.data_loader.search_claims(
            claim_type=claim_details.get('claim_type')
        )
        
        if similar_claims:
            avg_amount = sum(c['amount'] for c in similar_claims) / len(similar_claims)
            if claim_details.get('amount', 0) > avg_amount * 2:
                red_flags.append("Unusually high claim amount")
        
        # Check documentation completeness
        required_docs = self.get_required_documents(claim_details.get('claim_type'))
        provided_docs = claim_details.get('documents_provided', [])
        missing_docs = set(required_docs) - set(provided_docs)
        
        if missing_docs:
            red_flags.append(f"Missing required documents: {', '.join(missing_docs)}")
        
        return red_flags

    def get_required_documents(self, claim_type: str) -> List[str]:
        """Get list of required documents for a claim type"""
        document_requirements = {
            "Emergency Care": [
                "Hospital Report",
                "Medical Bills",
                "Treatment Records",
                "Emergency Room Documentation"
            ],
            "Prescription": [
                "Prescription",
                "Pharmacy Bill",
                "Doctor's Note"
            ],
            "Specialist Visit": [
                "Referral Letter",
                "Specialist Report",
                "Medical Bills",
                "Treatment Plan"
            ],
            "Preventive Care": [
                "Provider Report",
                "Medical Bills",
                "Preventive Care Schedule"
            ]
        }
        
        return document_requirements.get(claim_type, ["Medical Documentation", "Bills"])

    def suggest_settlement_amount(self, claim_details: Dict) -> float:
        """Suggest optimal settlement amount based on historical data"""
        similar_claims = self.data_loader.search_claims(
            claim_type=claim_details.get('claim_type'),
            status="Approved"
        )
        
        if not similar_claims:
            return claim_details.get('amount', 0) * 0.8  # Default 80% if no history
            
        avg_settlement_ratio = sum(
            c['settlement_amount'] / c['amount'] 
            for c in similar_claims 
            if c['settlement_amount'] is not None
        ) / len(similar_claims)
        
        return claim_details.get('amount', 0) * avg_settlement_ratio