# Squeeasy v0.2
# Developed by Chun-Ka Wong and Wing-Chun San
# wongeck@hku.hk
# Last updated: 31/01/2025

import litellm
import json
import pandas as pd
from pydantic import BaseModel, ValidationError, Field
from typing import List, Optional
import os
import re
from time import sleep
from itertools import combinations
import random
from openai import OpenAI
# litellm._turn_on_debug()

###############################################################################
# Settings
###############################################################################
os.environ['DEEPSEEK_API_KEY'] = ""
os.environ["OPENAI_API_KEY"] = ""

output_path = ""
var_scientific_question = ""

var_llm_agent1 = "o1-mini"
var_llm_Agent2 = "o1-mini"
var_llm_Agent3 = "o1-mini"

var_cycles = 10
var_retries = 3
var_matches = 1
var_max_pairs = 20
var_show_reasoning = False

CRITERIA = '''
1. Must be a new approach with no prior publications or report at all.
2. Out of the box thinking: discontinuity from existing ideas is encouraged.
3. Detailed experimental design with precision on what material to use.
4. Feasible, realistic and achievable with current technology.
5. You have to genuinely believe that it can solve the problem.
'''

###############################################################################
# Prompts
###############################################################################

LLM_AGENT_1_PROMPT = '''
Propose [ 1 ] break through method for the scientific question: {var_scientific_question}

**Criteria:
{CRITERIA}

**Thinking Process:
1. Exhaustively review existing approaches and ideas
2. Generate 5 break through ideas
3. Evaluate conceptual discontinuity from existing ideas
4. Harshly critique all ideas
5. Select top 1 idea
6. Meticulously develop experimental method
7. Harshly critique method flaws
8. Refine method to address all critiques
9. Make draft of JSON output
10. Review if JSON output fits the following description
11. Output final JSON output if it is correct

**Required JSON Format (strictly follow the format):
**Do not include special characters in JSON output**
**Do not include new lines in JSON output**
{{
  "ideas_considered_but_not_chosen": ["Idea 1", "Idea 2"], #Not to return "Idea 1" or "Idea 2"; it is just a placeholder. Return the banned ideas instead.
  "idea_title": "New Title",
  "idea_method": "Detailed Steps..."
}}
'''

LLM_AGENT_2_PROMPT = '''
Propose [ 1 ] break through method for the scientific question: {var_scientific_question}

**Banned ideas (STRICTLY PROHIBITED):
{var_discarded_ideas}

**Requirements:
{CRITERIA}

**Thinking Process:
1. Review banned ideas aloud
2. Exhaustively review existing approaches and ideas
3. Generate 5 break through ideas
4. Evaluate conceptual discontinuity from existing ideas
5. Harshly critique all ideas
6. Select top 1 idea
7. Meticulously develop experimental method
8. Harshly critique method flaws
9. Refine method to address all critiques
10. Make draft of JSON output
11. Review if JSON output fits the following description
12. Output final JSON output if it is correct

**Required JSON Output (strictly follow the format):
**Do not include special characters in JSON output**
**Do not include new lines in JSON output**
{{
  "ideas_considered_but_not_chosen": ["Idea 1", "Idea 2"], #Not to return "Idea 1" or "Idea 2"; it is just a placeholder. Return the banned ideas instead.
  "idea_title": "New Title",
  "idea_method": "Detailed Steps..."
}}
'''

LLM_AGENT_3_PROMPT = '''
Select the BEST idea for the scientific question: {var_scientific_question}

**Evaluation Criteria:
{CRITERIA}

**Ideas:
1. {var_title_1}: {var_method_1}
2. {var_title_2}: {var_method_2}

**Analysis Steps:
1. Review our evaluation criteria
2. Harshly critique both ideas
3. Select superior idea
4. Make draft of JSON output
5. Review if JSON output fits the following description
6. Output final JSON output if it is correct

**Required JSON Output (strictly follow the format):
**Do not include special characters in JSON output**
**Do not include new lines in JSON output**
{{
  "keep": "Chosen Title",
  "reject": "Rejected Title"
}}
'''

###############################################################################
# Pydantic Models for Response Validation
###############################################################################
class Agent1Response(BaseModel):
    ideas_considered_but_not_chosen: List[str] = Field(alias="ideas_considered_but_not_chosen")
    idea_title: str = Field(alias="idea_title")
    idea_method: str = Field(alias="idea_method")

class Agent2Response(BaseModel):
    ideas_considered_but_not_chosen: List[str] = Field(alias="ideas_considered_but_not_chosen")
    idea_title: str = Field(alias="idea_title")
    idea_method: str = Field(alias="idea_method")

class Agent3Response(BaseModel):
    keep: str = Field(alias="keep")
    reject: str = Field(alias="reject")

###############################################################################
# Utility Functions
###############################################################################
def update_elo(r1: float, r2: float, outcome: int, k: int = 32) -> tuple:
    e1 = 1 / (1 + 10 ** ((r2 - r1) / 400))
    e2 = 1 - e1
    return r1 + k * (outcome - e1), r2 + k * ((1 - outcome) - e2)

def robust_json_extractor(text: str) -> Optional[str]:
    try:
        json_match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
        if json_match:
            candidate = json_match.group()
            json.loads(candidate)
            return candidate
        return None
    except json.JSONDecodeError as e:
        print(f"JSON extraction error: {e}")
        return None

def execute_agent(
    prompt: str, 
    model: str, 
    validator: BaseModel,
    retries: int = 3,
    delay: float = 2.0
) -> Optional[BaseModel]:
    for attempt in range(1, retries + 1):
        try:
            print(f"Attempt {attempt}/{retries} - Model: {model}")
            response = litellm.completion(
                model=model,
                messages=[{"content": prompt, "role": "user"}],
            )
            content = response.choices[0].message.content

            if model == "deepseek-reasoner" and var_show_reasoning:
                print(response.choices[0].message.reasoning_content)

            json_str = robust_json_extractor(content)
            if json_str:
                parsed = validator.model_validate_json(json_str)
                return parsed

            raise ValueError("No valid JSON found in the response.")
        
        except Exception as e:
            print(f"Attempt {attempt} failed: {str(e)}")
            if attempt < retries:
                sleep(delay * attempt)
            else:
                print(f"execute_agent: exhausted all {retries} attempts; returning None.")
                return None

###############################################################################
# Agents
###############################################################################
def agent1_initial(question: str):
    prompt = LLM_AGENT_1_PROMPT.format(var_scientific_question=question, CRITERIA=CRITERIA)
    response = execute_agent(prompt, var_llm_agent1, Agent1Response, retries=var_retries)
    if response is not None:
        return (
            response.ideas_considered_but_not_chosen,
            response.idea_title,
            response.idea_method
        )
    return [], None, None

def Agent2_continuation(question: str, banned: list):
    prompt = LLM_AGENT_2_PROMPT.format(
        var_scientific_question=question,
        var_discarded_ideas=json.dumps(banned, indent=2),
        CRITERIA=CRITERIA
    )

    response = execute_agent(prompt, var_llm_Agent2, Agent2Response, retries=var_retries)
    if response is not None:
        return response.ideas_considered_but_not_chosen, response.idea_title, response.idea_method
    return [], None, None

def Agent3_evaluator(question: str, idea_a: tuple, idea_b: tuple):
    title_a, method_a = idea_a
    title_b, method_b = idea_b

    prompt = LLM_AGENT_3_PROMPT.format(
        var_scientific_question=question,
        var_title_1=title_a,
        var_method_1=method_a,
        var_title_2=title_b,
        var_method_2=method_b,
        CRITERIA=CRITERIA
    )

    response = execute_agent(prompt, var_llm_Agent3, Agent3Response, retries=var_retries)
    if response is not None:
        return response.keep, response.reject
    return None, None

###############################################################################
# Main Workflow
###############################################################################
def research_workflow(
    question: str, 
    cycles: int, 
    matches: int, 
    max_pairs: int
):
    all_ideas = []
    banned_ideas = []
    
    print("\n" + "═"*50 + "\n🔬 SQUEEASY WORKFLOW INITIATED\n" + "═"*50)
    print(f"Research Question: {question}")
    print(f"\nTotal ideas to Generate: {cycles}")
    print(f"Max Pairs to Evaluate: {max_pairs}")
    print(f"Pairwise Matches per Pair: {matches}")

    # --- PHASE 1: idea GENERATION ---
    print("\n" + "═"*50 + "\n🚀 PHASE 1: IDEAS GENERATION\n" + "═"*50)

    print(f"## Cycle 1: Generating idea ##")
    initial_banned, c_title, c_method = agent1_initial(question)
    if c_title is None:
        print("Agent1 failed to produce valid initial idea after all retries.")
        return pd.DataFrame()

    all_ideas.append({'title': c_title, 'method': c_method})
    # banned_ideas = [c_title] + initial_banned
    banned_ideas = [c_title] 
    print(f"\nCycle 1: Idea: {c_title}")
    print(f"\nCycle 1: Method: {c_method}")
    print(f"\nCycle 1: Ideas not selected in this cycle: {banned_ideas}")

    for cycle in range(2, cycles+1):
        print(f"\n## Cycle {cycle}: Generating idea ##")
        new_banned, new_title, new_method = Agent2_continuation(question, banned_ideas)
        if new_title is None:
            print(f"Agent2 failed on cycle {cycle}, skipping new idea this round.")
            continue

        all_ideas.append({'title': new_title, 'method': new_method})
        # banned_ideas += [new_title] + new_banned
        banned_ideas += [new_title]
        print(f"\nCycle {cycle}: Idea - {new_title}")
        print(f"\nCycle {cycle}: Method: {new_method}")
        print(f"\nCycle {cycle}: Ideas not selected in this prompt: {banned_ideas}")

    # --- PHASE 2: SUBSAMPLED PAIRWISE EVALUATION ---
    print("\n" + "═"*50 + "\n🔍 PHASE 2: SUBSAMPLED PAIRWISE EVALUATION\n" + "═"*50)

    elo_ideas = [
        {'title': c['title'], 'method': c['method'], 'elo': 1200} 
        for c in all_ideas
    ]

    random.seed(42)
    all_possible_pairs = list(combinations(elo_ideas, 2))
    if len(all_possible_pairs) > max_pairs:
        idea_pairs = random.sample(all_possible_pairs, max_pairs)
    else:
        idea_pairs = all_possible_pairs

    print(f"Evaluating {len(idea_pairs)} pairs (sampled from {len(all_possible_pairs)}) across {matches} matches")

    for match_num in range(1, matches + 1):
        print(f"\nMatch Round {match_num}/{matches}")
        for idx, (a, b) in enumerate(idea_pairs):
            kept_title, rejected_title = Agent3_evaluator(
                question=question,
                idea_a=(a['title'], a['method']),
                idea_b=(b['title'], b['method'])
            )

            if kept_title is None:
                print(f"⚠️ Agent3 failed for pair {idx+1}, skipping this match.")
                continue

            if kept_title == a['title']:
                winner, loser = a, b
            else:
                winner, loser = b, a

            new_winner_elo, new_loser_elo = update_elo(winner['elo'], loser['elo'], 1)
            winner['elo'] = new_winner_elo
            loser['elo'] = new_loser_elo

            print(f"Pair {idx+1}: {a['title'][:15]} vs {b['title'][:15]} => "
                  f"Winner: {kept_title[:15]} (New Elo: {winner['elo']:.1f}|{loser['elo']:.1f})")

    if not elo_ideas:
        print("No valid ideas were generated.")
        return pd.DataFrame()

    elo_ideas.sort(key=lambda x: x['elo'], reverse=True)

    results = pd.DataFrame([
        {
            'Rank': i + 1,
            'Title': c['title'],
            'Method': c['method'],
            'Elo_Rating': round(c['elo'], 1),
            'Method_Summary': (c['method'][:150] + '...') if len(c['method']) > 150 else c['method']
        }
        for i, c in enumerate(elo_ideas)
    ])
    return results

###############################################################################
# Main Execution
###############################################################################
if __name__ == "__main__":
    print("\n" + "═"*50 + "\n🚀 Starting the Squeeasy Workflow\n" + "═"*50)
    results = research_workflow(
        question=var_scientific_question,
        cycles=var_cycles,
        matches=var_matches,
        max_pairs=var_max_pairs
    )

    if not results.empty:
        results.to_csv(output_path, index=False)
        print("\n" + "═"*50 + "\n🏆 FINAL LEADERBOARD\n" + "═"*50)
        print(results[['Rank', 'Title', 'Elo_Rating', 'Method_Summary']].to_markdown(index=False))
        print(f"\nResults saved to: {output_path}")
    else:
        print("❌ Workflow failed to generate meaningful results")
