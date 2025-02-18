<h1>Squeeasy - Squeezing Scientific Creativity Out of Large Language Models Made Easy</h1>

<ul>
    <li><strong>Version:</strong> v0.4</li>
    <li><strong>Date:</strong> February 3, 2025</li>
    <li><strong>Developers:</strong> Chun-Ka Wong, Wing-Chun San</li>
    <li><strong>Contact:</strong> wongeck@hku.hk</li>
</ul>

<h2>Introduction</h2>
<p>Large language models (LLM) are increasingly capable of solving language and reasoning tasks. However, using LLM to generate scientific ideas is still challenging.</p>

<p>Squeeasy is a tool designed to streamline the process of generating creative scientific ideas using LLM and simplistic agents. The process involves following steps:</p>

<ul>
    <li>Review scientific literature for ideas that are already published using Perplexity API (Agent 3 - new).</li>
    <li>These published ideas will be saved in memory and will be excluded from subsequent idea generation.</li>
    <li>Propose multiple ideas given a scientific question (Agent 1).</li>
    <li>Critique and save the best idea among the proposed options.</li>
    <li>The newly generated idea will be saved in memory and will be excluded from subsequent idea generation.</li>
    <li>The process of idea generation will be repeated until the required number of ideas is generated.</li>
    <li>After generating the required number of ideas, 1 vs 1 comparison between the selected ideas will be performed using a LLM judge (Agent 2).</li>
    <li>Elo rating will be calculated to determine ranking of the ideas.</li>
    <li>Results will be saved as .html and .csv files.</li>
</ul>

<p>Observations we made with our preliminary experiments:</p>

<ul>
    <li>Reasoning models, such as DeepSeek-R1, OpenAI o1, and OpenAI o3-mini, give better performance than non-reasoning models.</li>
    <li>Only large LLM models allow generation of useful scientific ideas, such as DeepSeek-R1, OpenAI o1, and OpenAI o3-mini. The ideas generated by 70B or smaller models for Agents 1 are generally not satisfactory.</li>
    <li>It is necessary to fine tune the prompts, especially "CRITERIA", which is used to evaluate quality of the ideas generated.</li>
</ul>

<h2>Installation</h2>

<pre><code>git clone https://www.github.com/wongchunka/squeeasy
cd squeeasy
pip install -r requirements.txt
</code></pre>

<h2>Settings</h2>

<p>Before running Squeeasy, you must edit run.py to set the following parameters:</p>

<ul>
    <li>Squeeasy uses LiteLLM by default. Please visit <a href="https://github.com/BerriAI/litellm">LiteLLM repository</a> for instructions of setting API keys and LLM model names.</li>
    <li><code>output_path</code>: The path to save the results.</li>
    <li><code>var_litearture_review</code>: The query for performing literature review.</li>
    <li><code>var_scientific_question</code>: The scientific question to be used for generating ideas.</li>
    <li><code>var_llm_agent1</code>: The LLM model to be used for generating ideas.</li>
    <li><code>var_llm_Agent2</code>: The LLM model to be used for evaluating ideas in 1 vs 1 matches.</li>
    <li><code>var_llm_Agent3</code>: The LLM model to be used for performing literature review. Only perplexity/sonar-reasoning, perplexity/sonar-pro, and perplexirty/sonar are supported.</li>
    <li><code>var_cycles</code>: The number of ideas to be generated.</li>
    <li><code>var_retries</code>: The number of retries to be attempted if LLM call failed, such as due to API error or JSON parsing problem.</li>
    <li><code>var_matches</code>: The number of matches to be used for judging the ideas.</li>
    <li><code>var_max_pairs</code>: The maximum number of pairs to be used for judging the ideas.</li>
    <li><code>var_show_reasoning</code>: Whether to show the reasoning of the ideas (applicable to open source models only, such as DeepSeek-R1).</li>
    <li><code>var_literature_review</code>: Whether to perform literature review with Perplexity API.</li>
</ul>

<h2>Run Squeeasy on command line interface (CLI)</h2>

<pre><code>python run.py</code></pre>