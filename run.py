# Squeeasy v0.4
# Developed by Chun-Ka Wong and Wing-Chun San
# wongeck@hku.hk
# Last updated: 03/02/2025

import litellm
from litellm import completion
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
import concurrent.futures
import threading
# litellm._turn_on_debug()

###############################################################################
# Settings
###############################################################################
os.environ['DEEPSEEK_API_KEY'] = ""
os.environ["OPENAI_API_KEY"] = ""
os.environ['PERPLEXITYAI_API_KEY'] = ""

output_path = ""

var_litearture_review = ""

var_scientific_question = ""

var_llm_agent1 = "o3-mini"
var_llm_agent2 = "o3-mini"
var_llm_agent3 = "perplexity/sonar-reasoning"  # options: perplexity/sonar-reasoning, perplexity/sonar-pro, perplexity/sonar

var_cycles = 5
var_retries = 3
var_matches = 1
var_max_pairs = 10
var_show_reasoning = True  # Only applicable if deepseek/deepseek-reasoner is used
var_literature_review = True  # Set to True if perplexity API is available for literature review

CRITERIA = '''
1. Must be a new approach with no prior publications or report at all.
2. Out of the box thinking: discontinuity from existing ideas is encouraged.
3. Radical ideas are preferred.
4. Detailed experimental design with precision on what material to use.
5. Feasible, realistic and achievable with current technology.
6. You have to genuinely believe that it can solve the problem.
'''

###############################################################################
# Prompts
###############################################################################
LLM_agent_1_PROMPT = '''
Propose [ 1 ] break through method for the scientific question: {var_scientific_question}

**Banned ideas (STRICTLY PROHIBITED:
{var_discarded_ideas}

**Requirements:
{CRITERIA}

**Thinking Process:
0. Use different angles to see and think about the problem
1. Review banned ideas aloud
2. Exhaustively review previously reported approaches and ideas by others
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
  "ideas_considered_but_not_chosen": ["Idea 1", "Idea 2"],
  "idea_title": "New Title",
  "idea_method": "Detailed Steps..."
}}
'''

LLM_agent_2_PROMPT = '''
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

LLM_agent_3_PROMPT = '''
List all methods from the scientific litearture for the following scientific question.
In your answer, please be precise and specific about details, instead of giving over-simplified overview descriptions.
Scientific question to work on: 
{var_litearture_review}
'''

###############################################################################
# Pydantic Models for Response Validation
###############################################################################
class agent1Response(BaseModel):
    ideas_considered_but_not_chosen: List[str] = Field(alias="ideas_considered_but_not_chosen")
    idea_title: str = Field(alias="idea_title")
    idea_method: str = Field(alias="idea_method")

class agent2Response(BaseModel):
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
            print(prompt)
            response = litellm.completion(
                model=model,
                messages=[{"content": prompt, "role": "user"}],
            )
            content = response.choices[0].message.content

            if model == "deepseek/deepseek-reasoner" and var_show_reasoning:
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
def aagent1_brainstorm(question: str, banned: list):
    prompt = LLM_agent_1_PROMPT.format(
        var_scientific_question=question,
        var_discarded_ideas=json.dumps(banned, indent=2),
        CRITERIA=CRITERIA
    )

    response = execute_agent(prompt, var_llm_agent1, agent1Response, retries=var_retries)
    if response is not None:
        return response.ideas_considered_but_not_chosen, response.idea_title, response.idea_method
    return [], None, None

def agent2_evaluator(question: str, idea_a: tuple, idea_b: tuple):
    title_a, method_a = idea_a
    title_b, method_b = idea_b

    prompt = LLM_agent_2_PROMPT.format(
        var_scientific_question=question,
        var_title_1=title_a,
        var_method_1=method_a,
        var_title_2=title_b,
        var_method_2=method_b,
        CRITERIA=CRITERIA
    )

    response = execute_agent(prompt, var_llm_agent2, agent2Response, retries=var_retries)
    if response is not None:
        return response.keep, response.reject
    return None, None

def agent3_literature_review(question: str):
    prompt = LLM_agent_3_PROMPT.format(
        var_litearture_review=question,
    )
    response = completion(
        model=var_llm_agent3,
        messages=[{"content": prompt, "role": "user"}],
        stream=False
    )
    output_with_reasoning = response.choices[0].message.content
    output_without_reasoning = re.sub(r'<think>.*?</think>', '', output_with_reasoning, flags=re.DOTALL)
    output_without_reasoning = re.sub(r'<think>|</think>', '', output_without_reasoning)
    return output_without_reasoning

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
    literature_review_result = None  # To store literature review output
    print("\n" + "═"*50 + "\n🔬 SQUEEASY WORKFLOW INITIATED\n" + "═"*50)
    print(f"\nLiterature Review Query: {var_litearture_review}")
    print(f"\nResearch Question: {question}")
    print(f"\nTotal ideas to Generate: {cycles}")
    print(f"Max Pairs to Evaluate: {max_pairs}")
    print(f"Pairwise Matches per Pair: {matches}")

    # --- PHASE 0: LITERATURE REVIEW ---
    print("\n" + "═"*50 + "\n🚀 PHASE 0: LITERATURE REVIEW\n" + "═"*50)
    if var_literature_review:
        # Run the literature review LLM call in a separate thread
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_lit = executor.submit(agent3_literature_review, var_litearture_review)
            literature_review_result = future_lit.result()
        print(literature_review_result)
        banned_ideas += [literature_review_result]
    else:
        print("Literature review is not performed.")

    # --- PHASE 1: IDEAS GENERATION ---
    print("\n" + "═"*50 + "\n🚀 PHASE 1: IDEAS GENERATION\n" + "═"*50)
    agent1_cycle_outputs = []
    for cycle in range(1, cycles+1):
        print(f"\n## Cycle {cycle}: Generating idea ##")
        new_banned, new_title, new_method = aagent1_brainstorm(question, banned_ideas)
        if new_title is None:
            print(f"agent1 failed on cycle {cycle}, skipping new idea this round.")
            continue

        all_ideas.append({'title': new_title, 'method': new_method})
        banned_ideas += [new_title]
        print(f"\nCycle {cycle}: Idea - {new_title}")
        print(f"\nCycle {cycle}: Method: {new_method}")
        print(f"\nCycle {cycle}: Ideas not selected in this prompt: {banned_ideas}")
        # Save the raw output from agent1 for this cycle.
        agent1_cycle_outputs.append({
            "Cycle": cycle,
            "Banned_Ideas": new_banned,
            "Idea_Title": new_title,
            "Idea_Method": new_method
        })

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

    # Create a lock to update Elo scores safely
    elo_lock = threading.Lock()

    # Define a helper function to evaluate a single pair
    def evaluate_pair(a, b, pair_idx):
        kept_title, rejected_title = agent2_evaluator(
            question=question,
            idea_a=(a['title'], a['method']),
            idea_b=(b['title'], b['method'])
        )
        if kept_title is None:
            print(f"⚠️ agent2 failed for pair {pair_idx}, skipping this match.")
            return
        if kept_title == a['title']:
            winner, loser = a, b
        else:
            winner, loser = b, a
        new_winner_elo, new_loser_elo = update_elo(winner['elo'], loser['elo'], 1)
        with elo_lock:
            winner['elo'] = new_winner_elo
            loser['elo'] = new_loser_elo
        print(f"Pair {pair_idx}: {a['title'][:15]} vs {b['title'][:15]} => Winner: {kept_title[:15]} (New Elo: {winner['elo']:.1f}|{loser['elo']:.1f})")

    for match_num in range(1, matches + 1):
        print(f"\nMatch Round {match_num}/{matches}")
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = []
            for idx, (a, b) in enumerate(idea_pairs, start=1):
                futures.append(executor.submit(evaluate_pair, a, b, idx))
            # Wait for all pair evaluations to complete before proceeding to the next match round
            concurrent.futures.wait(futures)

    if not elo_ideas:
        print("No valid ideas were generated.")
        return pd.DataFrame(), agent1_cycle_outputs, literature_review_result

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
    return results, agent1_cycle_outputs, literature_review_result

###############################################################################
# Main Execution
###############################################################################
if __name__ == "__main__":
    print("\n" + "═"*50 + "\n🚀 Starting the Squeeasy Workflow\n" + "═"*50)
    results, agent1_cycle_outputs, literature_review_result = research_workflow(
        question=var_scientific_question,
        cycles=var_cycles,
        matches=var_matches,
        max_pairs=var_max_pairs
    )

    if not results.empty:
        # Save final leaderboard results.
        results.to_csv(output_path, index=False)
        print("\n" + "═"*50 + "\n🏆 FINAL LEADERBOARD\n" + "═"*50)
        print(results[['Rank', 'Title', 'Elo_Rating', 'Method_Summary']].to_markdown(index=False))
        print(f"\nResults saved to: {output_path}")
        
        # --- Separately save output from agent1 ---
        if agent1_cycle_outputs:
            df_agent1 = pd.DataFrame(agent1_cycle_outputs)
            # Convert the list of banned ideas into a JSON string for CSV compatibility.
            df_agent1['Banned_Ideas'] = df_agent1['Banned_Ideas'].apply(lambda x: json.dumps(x))
            agent1_output_path = output_path.replace('.csv', '_agent1.csv') if output_path else "agent1_output.csv"
            df_agent1.to_csv(agent1_output_path, index=False)
            print(f"Agent1 outputs saved to: {agent1_output_path}")
        
        # --- TASK 1: Generate HTML Visualization with Settings ---
        output_path_html = output_path.replace('.csv', '.html')
        html_content = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>Squeeasy Final Leaderboard</title>
<style>
    body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f9f9f9; }}
    h1, h2 {{ color: #333; }}
    table {{ border-collapse: collapse; width: 100%; margin-bottom: 40px; }}
    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
    th {{ background-color: #f2f2f2; }}
    .idea-section {{ margin-top: 40px; }}
    .idea {{ margin-bottom: 20px; padding: 10px; background-color: #fff; border: 1px solid #ddd; border-radius: 5px; }}
    .idea-title {{ font-weight: bold; font-size: 1.2em; margin-bottom: 5px; }}
    .idea-method {{ margin-left: 20px; white-space: pre-wrap; }}
    .settings-table {{ margin-bottom: 40px; }}
    .literature-review {{
        margin-bottom: 40px;
        padding: 10px;
        background-color: #eef;
        border: 1px solid #ccd;
        border-radius: 5px;
        overflow-x: auto;
        white-space: pre-wrap;
        word-wrap: break-word;
    }}
</style>
</head>
<body>
<h1>Squeeasy Final Leaderboard</h1>
<h2>Version 0.4 (03/02/2025)</h2>
<h2>Elo Table</h2>
<table>
  <thead>
    <tr>
      <th>Rank</th>
      <th>Title</th>
      <th>Elo Rating</th>
    </tr>
  </thead>
  <tbody>
"""
        # Add table rows for the Elo table
        for index, row in results.iterrows():
            html_content += f"<tr><td>{row['Rank']}</td><td>{row['Title']}</td><td>{row['Elo_Rating']}</td></tr>"
        html_content += """
  </tbody>
</table>
<h2>Settings</h2>
<table class="settings-table">
  <thead>
    <tr>
      <th>Parameter</th>
      <th>Value</th>
    </tr>
  </thead>
  <tbody>
    <tr><td>Research Question</td><td>""" + var_scientific_question.replace("\n", "<br>") + """</td></tr>
    <tr><td>Literature Review Query</td><td>""" + var_litearture_review.replace("\n", "<br>") + """</td></tr>
    <tr><td>LLM Agent 1 (Idea Generator)</td><td>""" + var_llm_agent1 + """</td></tr>
    <tr><td>LLM Agent 2 (Evaluator)</td><td>""" + var_llm_agent2 + """</td></tr>
    <tr><td>LLM Agent 3 (Literature Review)</td><td>""" + var_llm_agent3 + """</td></tr>
    <tr><td>Cycles</td><td>""" + str(var_cycles) + """</td></tr>
    <tr><td>Retries</td><td>""" + str(var_retries) + """</td></tr>
    <tr><td>Matches</td><td>""" + str(var_matches) + """</td></tr>
    <tr><td>Max Pairs</td><td>""" + str(var_max_pairs) + """</td></tr>
    <tr><td>Show Reasoning</td><td>""" + str(var_show_reasoning) + """</td></tr>
  </tbody>
</table>
<h2>Literature Review Results</h2>
<div class="literature-review">
    <pre>""" + (literature_review_result if literature_review_result is not None else "Literature review not performed.") + """</pre>
</div>
<div class="idea-section">
<h2>Idea Details</h2>
"""
        # Add detailed sections for each idea (Title and full Method)
        for index, row in results.iterrows():
            html_content += f"<div class='idea'><div class='idea-title'>Rank {row['Rank']}: {row['Title']}</div><div class='idea-method'>{row['Method']}</div></div>"
        html_content += """
</div>
</body>
</html>
"""
        with open(output_path_html, "w", encoding="utf-8") as f:
            f.write(html_content)
        print(f"HTML visualization saved to: {output_path_html}")
    else:
        print("❌ Workflow failed to generate meaningful results")
