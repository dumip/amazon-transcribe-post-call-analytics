"""
GenAI Analytics module for PCA - provides sentiment analysis, entity extraction, 
PII detection, and category assignment using Amazon Bedrock for unsupported languages.

Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
SPDX-License-Identifier: Apache-2.0
"""
import boto3
import json
import time
import os
import pcaconfiguration as cf
from botocore.exceptions import ClientError


class GenAIAnalytics:
    """
    GenAI Analytics class providing sentiment analysis, entity extraction,
    PII detection, and category assignment using Amazon Bedrock.
    """
    
    def __init__(self):
        self.bedrock_client = boto3.client('bedrock-runtime')
        self.ssm_client = boto3.client('ssm')
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        
    def get_token_consumption(self):
        """
        Returns total input/output tokens consumed by this instance.
        
        Returns:
            dict: Dictionary with 'input_tokens' and 'output_tokens' counts
        """
        consumption = {
            'input_tokens': self.total_input_tokens,
            'output_tokens': self.total_output_tokens
        }
        print(f"Total Bedrock token consumption - Input: {self.total_input_tokens}, Output: {self.total_output_tokens}")
        return consumption
    
    def _get_prompt_parameter(self, prompt_type, language, parameter_type, default_fallback=None):
        """
        Get a prompt parameter from SSM Parameter Store.
        
        Args:
            prompt_type: Type of prompt (sentiment-analysis, entity-extraction, etc.)
            language: Language code (e.g., 'ro', 'de', 'fr')
            parameter_type: Type of parameter ('version' or 'identifier')
            default_fallback: Default value to return if parameter not found
            
        Returns:
            str: Parameter value
        """
        try:
            # Use CloudFormation stack name in parameter name to match template structure
            stack_name = cf.STACK_NAME
            # Use consistent naming convention: pca-{prompt_type}-prompt-{parameter_type}-{language}
            parameter_name = f"{stack_name}-pca-{prompt_type}-prompt-{parameter_type}-{language}"
            
            response = self.ssm_client.get_parameter(Name=parameter_name)
            return response['Parameter']['Value']
        except ClientError:
            # Fallback to default language if specific language not found
            try:
                parameter_name = f"{stack_name}-pca-{prompt_type}-prompt-{parameter_type}-default"
                response = self.ssm_client.get_parameter(Name=parameter_name)
                return response['Parameter']['Value']
            except ClientError:
                if default_fallback is not None:
                    return default_fallback
                raise Exception(f"No prompt {parameter_type} found for {prompt_type} in language {language} or default")

    def _get_prompt_version(self, prompt_type, language):
        """
        Get the prompt version from SSM Parameter Store.
        
        Args:
            prompt_type: Type of prompt (sentiment-analysis, entity-extraction, etc.)
            language: Language code (e.g., 'ro', 'de', 'fr')
            
        Returns:
            str: Prompt version to use (defaults to '1' if not found)
        """
        return self._get_prompt_parameter(prompt_type, language, "version", "1")

    def _get_prompt_identifier(self, prompt_type, language):
        """
        Get the prompt identifier (ARN) from SSM Parameter Store.
        
        Args:
            prompt_type: Type of prompt (sentiment-analysis, entity-extraction, etc.)
            language: Language code (e.g., 'ro', 'de', 'fr')
            
        Returns:
            str: Prompt ARN identifier
        """
        return self._get_prompt_parameter(prompt_type, language, "identifier")

    
    def _invoke_bedrock_prompt(self, prompt_arn, variables):
        """
        Invoke Bedrock using a prompt ARN with variables using the Converse API.
        
        Args:
            prompt_arn: ARN of the prompt version from Bedrock Prompt Management
            variables: Dictionary of variables to substitute in the prompt
            
        Returns:
            dict: Response from the model
        """
        try:
            print(f"Invoking Bedrock prompt: {prompt_arn}")
            
            # Convert variables to the format expected by Converse API
            prompt_variables = {}
            for key, value in variables.items():
                prompt_variables[key] = {"text": str(value)}
            
            # Use Converse API with prompt ARN and variables
            response = self.bedrock_client.converse(
                modelId=prompt_arn,
                promptVariables=prompt_variables,
                messages=[]  # Empty messages since we're using a prompt
            )
            
            # Track and log token usage
            if 'usage' in response:
                input_tokens = response['usage'].get('inputTokens', 0)
                output_tokens = response['usage'].get('outputTokens', 0)
                self.total_input_tokens += input_tokens
                self.total_output_tokens += output_tokens
                print(f"Bedrock token usage - Input: {input_tokens}, Output: {output_tokens}")
            
            print(f"Bedrock prompt invocation completed successfully")
            return response
            
        except Exception as e:
            print(f"Error invoking Bedrock prompt {prompt_arn}: {str(e)}")
            raise e
    
    def _get_prompt_arn(self, prompt_type, language):
        """
        Get the complete prompt ARN including version from SSM Parameter Store.
        
        Args:
            prompt_type: Type of prompt (sentiment-analysis, entity-extraction, etc.)
            language: Language code (e.g., 'ro', 'de', 'fr')
            
        Returns:
            str: Complete prompt ARN with version (e.g., arn:aws:bedrock:region:account:prompt/id:version)
        """
        try:
            prompt_identifier = self._get_prompt_identifier(prompt_type, language)
            prompt_version = self._get_prompt_version(prompt_type, language)
            
            # Construct the full ARN with version
            if ':' in prompt_version:
                # Version is already included in the ARN
                full_arn = prompt_identifier
            else:
                # Append version to the ARN
                full_arn = f"{prompt_identifier}:{prompt_version}"
            
            # Validate ARN format
            if not full_arn.startswith('arn:aws:bedrock:'):
                raise Exception(f"Invalid prompt ARN format: {full_arn}")
            
            if ':prompt/' not in full_arn:
                raise Exception(f"ARN does not contain ':prompt/' segment: {full_arn}")
            
            print(f"Retrieved prompt ARN for {prompt_type} ({language}): {full_arn}")
            return full_arn
                
        except Exception as e:
            print(f"Failed to get prompt ARN for {prompt_type} in language '{language}': {str(e)}")
            raise Exception(f"Cannot get prompt ARN for {prompt_type} in language '{language}': {str(e)}")
    
    def genai_sentiment_analysis(self, text, speaker, language_code):
        """
        Perform sentiment analysis using Amazon Bedrock.
        
        Args:
            text: Text to analyze for sentiment
            speaker: Speaker identifier
            language_code: Language code (e.g., 'ro' for Romanian)
            
        Returns:
            dict: Sentiment analysis results compatible with Comprehend format
        """
        try:
            print(f"Starting GenAI sentiment analysis for language: {language_code}")
            
            # Get the prompt ARN with version
            prompt_arn = self._get_prompt_arn("sentiment-analysis", language_code)
            
            # Prepare variables for the prompt
            variables = {
                "speaker": speaker,
                "segment_text": text
            }
            
            # Invoke the prompt directly using Bedrock Prompt Management
            response = self._invoke_bedrock_prompt(prompt_arn, variables)
            
            # Extract the JSON response from the model output (Converse API format)
            content = response['output']['message']['content'][0]['text']
            
            # Parse the JSON response
            try:
                # Extract JSON from the response (it might be wrapped in markdown)
                json_start = content.find('{')
                json_end = content.rfind('}') + 1
                if json_start != -1 and json_end != -1:
                    json_str = content[json_start:json_end]
                    sentiment_result = json.loads(json_str)
                else:
                    raise ValueError("No JSON found in response")
            except (json.JSONDecodeError, ValueError) as e:
                print(f"Error parsing GenAI sentiment response: {str(e)}")
                print(f"Raw response: {content}")
                # Return neutral sentiment as fallback
                return self._get_neutral_sentiment_response()
            
            # Convert to Comprehend-compatible format
            comprehend_response = {
                "Sentiment": sentiment_result.get("segmentSentiment", "NEUTRAL"),
                "SentimentScore": {
                    "Positive": sentiment_result.get("segmentPositive", 0.0),
                    "Negative": sentiment_result.get("segmentNegative", 0.0),
                    "Neutral": 5.0 - sentiment_result.get("segmentPositive", 0.0) - sentiment_result.get("segmentNegative", 0.0)
                }
            }
            
            print(f"GenAI sentiment analysis completed - Sentiment: {comprehend_response['Sentiment']}")
            return comprehend_response
            
        except Exception as e:
            print(f"Error in GenAI sentiment analysis for language {language_code}: {str(e)}")
            return self._get_neutral_sentiment_response()
    
    def _get_neutral_sentiment_response(self):
        """
        Returns a neutral sentiment response as fallback.
        
        Returns:
            dict: Neutral sentiment response in Comprehend format
        """
        return {
            "Sentiment": "NEUTRAL",
            "SentimentScore": {
                "Positive": 0.0,
                "Negative": 0.0,
                "Neutral": 5.0
            }
        }
    
    def genai_entity_extraction(self, text, speaker, language_code):
        """
        Perform entity extraction using Amazon Bedrock.
        
        Args:
            text: Text to analyze for entities
            speaker: Speaker identifier
            language_code: Language code
            
        Returns:
            dict: Entity extraction results compatible with Comprehend format
        """
        try:
            # Get the prompt ARN with version
            prompt_arn = self._get_prompt_arn("entity-extraction", language_code)
            
            # Prepare variables for the prompt
            variables = {
                "speaker": speaker,
                "segment_text": text
            }
            
            # Invoke the prompt directly using Bedrock Prompt Management
            response = self._invoke_bedrock_prompt(prompt_arn, variables)
            
            # TODO: Parse response and convert to Comprehend format
            # For now, return empty entities
            return {"Entities": []}
            
        except Exception as e:
            print(f"Error in GenAI entity extraction: {str(e)}")
            return {"Entities": []}
    
    def genai_pii_detection(self, text, speaker, language_code):
        """
        Perform PII detection using Amazon Bedrock.
        
        Args:
            text: Text to analyze for PII
            speaker: Speaker identifier
            language_code: Language code
            
        Returns:
            dict: PII detection results
        """
        try:
            # Get the prompt ARN with version
            prompt_arn = self._get_prompt_arn("pii-detection", language_code)
            
            # Prepare variables for the prompt
            variables = {
                "speaker": speaker,
                "segment_text": text
            }
            
            # Invoke the prompt directly using Bedrock Prompt Management
            response = self._invoke_bedrock_prompt(prompt_arn, variables)
            
            # TODO: Parse response and convert to expected format
            # For now, return no PII detected
            return {"piiEntities": [], "maskedSegmentText": text}
            
        except Exception as e:
            print(f"Error in GenAI PII detection: {str(e)}")
            return {"piiEntities": [], "maskedSegmentText": text}
    
    def genai_category_assignment(self, transcript, language_code):
        """
        Perform category assignment using Amazon Bedrock.
        
        Args:
            transcript: Full conversation transcript
            language_code: Language code
            
        Returns:
            dict: Category assignment results
        """
        try:
            # Get the prompt ARN with version
            prompt_arn = self._get_prompt_arn("category-assignment", language_code)
            
            # Prepare variables for the prompt
            variables = {
                "transcript": transcript
            }
            
            # Invoke the prompt directly using Bedrock Prompt Management
            response = self._invoke_bedrock_prompt(prompt_arn, variables)
            
            # TODO: Parse response and convert to expected format
            # For now, return default category assignment
            return {
                "primaryCategory": "GENERAL_INQUIRY",
                "primaryConfidence": 0.5,
                "secondaryCategories": [],
                "reasoning": "Default category assignment",
                "callOutcome": "UNRESOLVED",
                "customerSatisfaction": "NEUTRAL"
            }
            
        except Exception as e:
            print(f"Error in GenAI category assignment: {str(e)}")
            return {
                "primaryCategory": "GENERAL_INQUIRY",
                "primaryConfidence": 0.5,
                "secondaryCategories": [],
                "reasoning": "Default category assignment",
                "callOutcome": "UNRESOLVED",
                "customerSatisfaction": "NEUTRAL"
            }