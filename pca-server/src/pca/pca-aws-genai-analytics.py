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
        return {
            'input_tokens': self.total_input_tokens,
            'output_tokens': self.total_output_tokens
        }
    
    def _get_prompt_version(self, prompt_type, language):
        """
        Get the prompt version from SSM Parameter Store.
        
        Args:
            prompt_type: Type of prompt (sentiment-analysis, entity-extraction, etc.)
            language: Language code (e.g., 'ro', 'de', 'fr')
            
        Returns:
            str: Prompt version to use (defaults to 'DEFAULT' if not found)
        """
        try:
            # Use CloudFormation stack name in parameter name to match template structure
            stack_name = os.environ.get('STACK_NAME')
            # Use kebab-case naming convention: sentiment-analysis-prompt-version-{language}
            parameter_name = f"{stack_name}-{prompt_type}-prompt-version-{language}"
            
            response = self.ssm_client.get_parameter(Name=parameter_name)
            return response['Parameter']['Value']
        except ClientError:
            # Fallback to default version
            return "1"
    
    def _invoke_bedrock_model(self, prompt, model_id="anthropic.claude-3-5-haiku-20241022-v1:0"):
        """
        Invoke Bedrock model with the given prompt.
        
        Args:
            prompt: The prompt to send to the model
            model_id: The Bedrock model ID to use
            
        Returns:
            dict: Response from the model
        """
        try:
            # Prepare the request body for Claude
            request_body = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 1000,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            }
            
            # Invoke the model
            response = self.bedrock_client.invoke_model(
                modelId=model_id,
                body=json.dumps(request_body)
            )
            
            # Parse the response
            response_body = json.loads(response['body'].read())
            
            # Track token usage
            if 'usage' in response_body:
                self.total_input_tokens += response_body['usage'].get('input_tokens', 0)
                self.total_output_tokens += response_body['usage'].get('output_tokens', 0)
            
            return response_body
            
        except Exception as e:
            print(f"Error invoking Bedrock model: {str(e)}")
            raise e
    
    def _get_bedrock_prompt(self, prompt_name, version="DEFAULT"):
        """
        Get prompt from Amazon Bedrock Prompt Management.
        
        Args:
            prompt_name: Name of the prompt in Bedrock Prompt Management
            version: Version of the prompt to retrieve
            
        Returns:
            str: The prompt template
        """
        try:
            bedrock_agent_client = boto3.client('bedrock-agent')
            
            # Get the prompt from Bedrock Prompt Management
            response = bedrock_agent_client.get_prompt(
                promptIdentifier=prompt_name,
                promptVersion=version
            )
            
            # Extract the prompt text from the response
            variants = response['variants']
            if variants and len(variants) > 0:
                template_configuration = variants[0].get('templateConfiguration', {})
                if 'text' in template_configuration:
                    return template_configuration['text']['text']
            
            raise Exception(f"No prompt text found for {prompt_name}")
            
        except Exception as e:
            error_msg = f"Failed to retrieve prompt '{prompt_name}' version '{version}' from Bedrock Prompt Management: {str(e)}"
            print(f"Error: {error_msg}")
            raise Exception(error_msg)
    
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
            # Get the appropriate prompt version
            prompt_version = self._get_prompt_version("sentiment-analysis", language_code)
            
            # Get the prompt template from Bedrock Prompt Management
            # Use CloudFormation naming convention: {StackName}-pca-sentiment-analysis-{language}
            stack_name = os.environ.get('STACK_NAME')
            prompt_name = f"{stack_name}-pca-sentiment-analysis-{language_code}"
            try:
                prompt_template = self._get_bedrock_prompt(prompt_name, prompt_version)
            except Exception as e:
                raise Exception(f"Cannot perform GenAI sentiment analysis for language '{language_code}': {str(e)}")
            
            # Replace variables in the prompt
            prompt = prompt_template.replace("{{speaker}}", speaker)
            prompt = prompt.replace("{{segment_text}}", text)
            
            # Invoke the model
            response = self._invoke_bedrock_model(prompt)
            
            # Extract the JSON response from the model output
            content = response['content'][0]['text']
            
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
            
            return comprehend_response
            
        except Exception as e:
            print(f"Error in GenAI sentiment analysis: {str(e)}")
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
        # Placeholder implementation - to be implemented later
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
        # Placeholder implementation - to be implemented later
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
        # Placeholder implementation - to be implemented later
        return {
            "primaryCategory": "GENERAL_INQUIRY",
            "primaryConfidence": 0.5,
            "secondaryCategories": [],
            "reasoning": "Default category assignment",
            "callOutcome": "UNRESOLVED",
            "customerSatisfaction": "NEUTRAL"
        }