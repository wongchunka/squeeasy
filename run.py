# Squeeasy v0.1 by Chun-Ka Wong and Wing-Chun San
# Email: wongeck@hku.hk

import litellm
import json
import pandas as pd
from pydantic import BaseModel, ValidationError, Field
from typing import List, Optional
import os
import re
from time import sleep
# litellm._turn_on_debug()

# Setting
os.environ['DEEPSEEK_API_KEY'] = ""
output_path = ""
var_scientific_question = ""
var_cycles = 10
var_retries = 5

# Pydantic Models for Response Validation
class Agent1Response(BaseModel):
    ideas_considered_and_banned: List[str] = Field(..., alias="Ideas_considered_and_banned")
    concept_1_title: str = Field(..., alias="Concept_1_title")
    concept_1_method_summary: str = Field(..., alias="Concept_1_method_summary")
    concept_2_title: str = Field(..., alias="Concept_2_title")
    concept_2_method_summary: str = Field(..., alias="Concept_2_method_summary")

class Agent2Response(BaseModel):
    keep: str = Field(..., alias="Keep")
    reject: str = Field(..., alias="Reject")

class Agent3Response(BaseModel):
    discarded_concepts_add: List[str] = Field(..., alias="Ideas_considered_and_banned_new")
    concept_title: str = Field(..., alias="Concept_new_title")
    concept_method: str = Field(..., alias="Concept_new_method")

# Enhanced Prompts with JSON Examples
LLM_AGENT_1_PROMPT = '''
Propose [ 2 ] break through methods for {var_scientific_question}

**Criteria:**
1. First to report: No prior publications
2. Conceptual breakthrough: Complete discontinuity from existing concepts
3. Implementable: Detailed experimental steps

**Thinking Process:**
1. Analyze existing approaches
2. Generate 3 novel concepts
3. Evaluate conceptual discontinuity
4. Critique all concepts
5. Select top 2 concepts
6. Develop experimental methods
7. Identify method flaws
8. Refine methods addressing critiques
9. Make draft of JSON output
10. Review if JSON output fits the following description
11. Output final JSON output if it is correct

**Required JSON Format (strictly follow the format):**
**Do not include special characters in JSON output**
**Do not include new lines in JSON output**
{{
  "Ideas_considered_and_banned": ["idea1", "idea2"],
  "Concept_1_title": "Title1",
  "Concept_1_method_summary": "Step1...",
  "Concept_2_title": "Title2",
  "Concept_2_method_summary": "Step2..."
}}
'''

LLM_AGENT_2_PROMPT = '''
Select the BEST concept for {var_scientific_question}:

**Evaluation Criteria:**
1. Novelty: No prior art
2. Conceptual leap: Radical departure from existing
3. Feasibility: Clear implementation path

**Concepts:**
1. {var_title_1}: {var_method_1}
2. {var_title_2}: {var_method_2}

**Analysis Steps:**
1. Compare against criteria
2. Identify weaknesses in both
3. Select superior concept
4. Make draft of JSON output
5. Review if JSON output fits the following description
6. Output final JSON output if it is correct

**Required JSON Output (strictly follow the format):**
**Do not include special characters in JSON output**
**Do not include new lines in JSON output**
{{
  "Keep": "Chosen Title",
  "Reject": "Rejected Title"
}}
'''

LLM_AGENT_3_PROMPT = '''
Develop NEW concept for {var_scientific_question}

**Banned Ideas:**
{var_discarded_concepts}

**Requirements:**
1. Avoid banned concepts
2. True conceptual breakthrough
3. Detailed experimental plan

**Thinking Process:**
1. Review banned concepts
2. Generate 3 new ideas
3. Validate conceptual novelty
4. Critique all ideas
5. Select best concept
6. Develop method with flaw analysis
7. Refine method
8. Make draft of JSON output
9. Review if JSON output fits the following description
10. Output final JSON output if it is correct

**Required JSON Output (strictly follow the format):**
**Do not include special characters in JSON output**
**Do not include new lines in JSON output**
{{
  "Ideas_considered_and_banned_new": ["ideaA", "ideaB"],
  "Concept_new_title": "NewTitle",
  "Concept_new_method": "DetailedSteps..."
}}
'''

def robust_json_extractor(text: str) -> Optional[str]:
    """Extract first valid JSON block from text with error recovery"""
    try:
        json_match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
        if json_match:
            candidate = json_match.group()
            json.loads(candidate)  # Validate JSON
            return candidate
        return None
    except json.JSONDecodeError as e:
        print(f"JSON extraction error: {e}")
        return None

def execute_agent(prompt: str, model: str, validator: BaseModel, retries: int = 3, delay: float = 2.0):
    """Universal agent executor with enhanced error handling"""
    for attempt in range(1, retries + 1):
        try:
            response = litellm.completion(
                model=model,
                messages=[{"content": prompt, "role": "user"}],
                temperature=0.3 if attempt > 1 else 0.7
            )
            # print(response.choices[0].message.provider_specific_fields["reasoning_content"])
            content = response.choices[0].message.content
            print(f"Attempt {attempt} Raw Response:\n{content}\n{'-'*40}")
            if json_str := robust_json_extractor(content):
                parsed = validator.model_validate_json(json_str)
                return parsed
            raise ValueError("No valid JSON found")
        except (ValidationError, json.JSONDecodeError, ValueError) as e:
            print(f"Attempt {attempt} failed: {str(e)}")
            if attempt < retries:
                sleep(delay * attempt)  # Exponential backoff
            else:
                raise RuntimeError(f"Failed after {retries} attempts: {str(e)}")

def agent1_initial(question: str, retries: int = 3):
    """First-stage concept generation"""
    prompt = LLM_AGENT_1_PROMPT.format(var_scientific_question=question)
    print("\n" + "═"*50 + "\n🚀 Agent 1: Initial Concept Generation\n" + "═"*50)
    print(prompt)
    response = execute_agent(prompt, "deepseek/deepseek-reasoner", Agent1Response, retries)
    print(response)
    return (
        response.ideas_considered_and_banned,
        response.concept_1_title,
        response.concept_1_method_summary,
        response.concept_2_title,
        response.concept_2_method_summary
    )

def agent2_evaluator(question: str, concept_a: tuple, concept_b: tuple, retries: int = 3):
    """Concept selection agent"""
    title_a, method_a = concept_a
    title_b, method_b = concept_b
    prompt = LLM_AGENT_2_PROMPT.format(
        var_scientific_question=question,
        var_title_1=title_a,
        var_method_1=method_a,
        var_title_2=title_b,
        var_method_2=method_b
    )
    print("\n" + "═"*50 + "\n🔍 Agent 2: Concept Evaluation\n" + "═"*50)
    print(prompt)
    response = execute_agent(prompt, "deepseek/deepseek-reasoner", Agent2Response, retries)
    print(response)
    return response.keep, response.reject

def agent3_continuation(question: str, banned: list, retries: int = 3):
    """Continued concept development agent"""
    prompt = LLM_AGENT_3_PROMPT.format(
        var_scientific_question=question,
        var_discarded_concepts=json.dumps(banned, indent=2)
    )
    print("\n" + "═"*50 + "\n💡 Agent 3: Continuation Concept Generation\n" + "═"*50)
    print(prompt)
    response = execute_agent(prompt, "deepseek/deepseek-reasoner", Agent3Response, retries)
    print(response)
    return response.discarded_concepts_add, response.concept_title, response.concept_method

def research_workflow(question: str, cycles: int = 10, retries: int = 3):
    """Main research iteration loop"""
    df = pd.DataFrame(columns=[
        'Cycle', 'New_Concept', 'New_Method', 
        'Current_Best', 'Current_Method', 
        'Rejected', 'Banned_Concepts'
    ])
    try:
        banned, *concepts = agent1_initial(question, retries)
        kept_title, rejected_title = agent2_evaluator(question, 
            (concepts[0], concepts[1]), 
            (concepts[2], concepts[3]), retries)
        current_best = (kept_title, concepts[1] if kept_title == concepts[0] else concepts[3])
        banned += [rejected_title]
    except Exception as e:
        print(f"❌ Initialization failed: {e}")
        return df
    for cycle in range(1, cycles + 1):
        try:
            new_banned, new_title, new_method = agent3_continuation(question, banned, retries)
            banned += new_banned
            selected_title, rejected_title = agent2_evaluator(question, 
                current_best, (new_title, new_method), retries)
            if selected_title == new_title:
                current_best = (new_title, new_method) 
            df = pd.concat([df, pd.DataFrame([{
                'Cycle': cycle,
                'New_Concept': new_title,
                'New_Method': new_method,
                'Current_Best': current_best[0],
                'Current_Method': current_best[1],
                'Rejected': rejected_title,
                'Banned_Concepts': str(banned)
            }])], ignore_index=True)
            df.to_csv(output_path, index=False)
            print(f"✅ Cycle {cycle} completed - Current Best: {current_best[0]}")
        except Exception as e:
            print(f"⚠️ Cycle {cycle} failed: {e}")
            continue
    return df

# Execution Example
if __name__ == "__main__":
    if not os.environ['DEEPSEEK_API_KEY'] or not output_path or not var_scientific_question or not var_cycles or not var_retries:
        print("❌ Settings are not set. Please input settings in run.py.")
        exit()
    research_question = (var_scientific_question)
    results = research_workflow(
        question=research_question,
        cycles=var_cycles,
        retries=var_retries
    )
    print("\nFinal Results:")
    print(results[['Cycle', 'Current_Best', 'Rejected']].to_markdown(index=False))