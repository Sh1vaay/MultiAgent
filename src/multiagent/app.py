import chainlit as cl
from crewai import Agent, Task, Crew, LLM
from crewai.knowledge.source.pdf_knowledge_source import PDFKnowledgeSource
from crewai.knowledge.source.text_file_knowledge_source import TextFileKnowledgeSource
from langchain_community.tools import DuckDuckGoSearchRun
import os
import shutil
import json
import pandas as pd
from dotenv import load_dotenv, find_dotenv
import warnings
import yaml
from typing import List
from pydantic import BaseModel, Field

# Pydantic Models (same as before)
class TaskEstimate(BaseModel):
    task_name: str = Field(..., description="Name of the task")
    estimated_time_hours: float = Field(..., description="Estimated time to complete the task in hours")
    required_resources: List[str] = Field(..., description="List of resources required to complete the task")

class Milestone(BaseModel):
    milestone_name: str = Field(..., description="Name of the milestone")
    tasks: List[str] = Field(..., description="List of task IDs associated with this milestone")

class RiskEntry(BaseModel):
    risk_name: str = Field(..., description="Name of the project risk")
    severity: str = Field(..., description="Severity level: High, Medium, or Low")
    mitigation_strategy: str = Field(..., description="Strategy to mitigate this risk")

class ProjectPlan(BaseModel):
    tasks: List[TaskEstimate] = Field(..., description="List of tasks with their estimates")
    milestones: List[Milestone] = Field(..., description="List of project milestones")
    risks: List[RiskEntry] = Field(default=[], description="List of project risks and mitigations")

# Load environment variables and configurations
warnings.filterwarnings("ignore")
load_dotenv(find_dotenv())

current_dir = os.path.dirname(os.path.abspath(__file__))
files = {
    'agents': os.path.join(current_dir, 'config', 'agents.yaml'),
    'tasks': os.path.join(current_dir, 'config', 'tasks.yaml')
}
configs = {}
for config_type, file_path in files.items():
    with open(file_path, 'r', encoding='utf-8') as file:
        configs[config_type] = yaml.safe_load(file)

agents_config = configs['agents']
tasks_config = configs['tasks']

def get_llm(provider: str = None, model: str = None) -> LLM:
    """
    Get the configured CrewAI LLM instance based on provider and model.
    """
    if not provider:
        # Auto-detect from environment variables
        if os.environ.get("OPENAI_API_KEY"):
            provider = "openai"
        elif os.environ.get("GROQ_API_KEY"):
            provider = "groq"
        else:
            provider = "ollama"

    provider = provider.lower().strip()
    if provider == "openai":
        return LLM(
            model=model or os.environ.get("OPENAI_MODEL_NAME", "gpt-4o-mini"),
            api_key=os.environ.get("OPENAI_API_KEY")
        )
    elif provider == "groq":
        return LLM(
            model=model or os.environ.get("GROQ_MODEL_NAME", "groq/llama3-8b-8192"),
            api_key=os.environ.get("GROQ_API_KEY")
        )
    elif provider == "ollama":
        return LLM(
            model=model or os.environ.get("OLLAMA_MODEL", "ollama/llama3"),
            base_url=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
        )
    else:
        raise ValueError(f"Unknown provider: {provider}")

def get_knowledge_config(provider, file_path):
    knowledge_sources = []
    if file_path:
        os.makedirs('knowledge', exist_ok=True)
        basename = os.path.basename(file_path)
        dest_path = os.path.join('knowledge', basename)
        
        if os.path.abspath(file_path) != os.path.abspath(dest_path):
            shutil.copy(file_path, dest_path)
            
        if file_path.endswith('.pdf'):
            knowledge_sources.append(PDFKnowledgeSource(file_paths=[basename]))
        else:
            knowledge_sources.append(TextFileKnowledgeSource(file_paths=[basename]))

    # Configure embedder
    if provider == "openai":
        embedder = {
            "provider": "openai",
            "config": {
                "model": "text-embedding-3-small"
            }
        }
    else:
        embedder = {
            "provider": "huggingface",
            "config": {
                "model": "sentence-transformers/all-MiniLM-L6-v2"
            }
        }

    return knowledge_sources, embedder

def generate_dashboard_html(project_plan):
    tasks_html = ""
    for idx, t in enumerate(project_plan.tasks):
        resources_badges = "".join([f'<span class="badge">{r}</span>' for r in t.required_resources])
        tasks_html += f"""
        <tr>
            <td>T{idx+1}</td>
            <td class="task-name">{t.task_name}</td>
            <td>{t.estimated_time_hours} hrs</td>
            <td>{resources_badges}</td>
        </tr>
        """

    milestones_html = ""
    for idx, m in enumerate(project_plan.milestones):
        tasks_list = ", ".join(m.tasks)
        milestones_html += f"""
        <div class="milestone-card">
            <div class="milestone-header">
                <span class="milestone-number">M{idx+1}</span>
                <span class="milestone-title">{m.milestone_name}</span>
            </div>
            <div class="milestone-body">
                <p><strong>Associated Tasks:</strong> {tasks_list}</p>
            </div>
        </div>
        """

    # Interactive Gantt Chart Calculations
    current_time = 0.0
    gantt_rows = ""
    total_hours = sum([t.estimated_time_hours for t in project_plan.tasks])
    max_days = max(10.0, round(total_hours / 8.0) + 2)

    for idx, t in enumerate(project_plan.tasks):
        start_time = current_time
        end_time = start_time + t.estimated_time_hours
        current_time = end_time

        start_day = round(start_time / 8.0, 1)
        duration_days = round(t.estimated_time_hours / 8.0, 1)

        left_pct = (start_day / max_days) * 100
        width_pct = (duration_days / max_days) * 100
        width_pct = max(2.0, width_pct) # Minimum visibility width

        gantt_rows += f"""
        <div class="gantt-row">
            <div class="gantt-label">
                <span class="task-id">T{idx+1}</span>
                <span class="task-title" title="{t.task_name}">{t.task_name}</span>
            </div>
            <div class="gantt-track">
                <div class="gantt-bar" style="left: {left_pct}%; width: {width_pct}%;">
                    <span class="gantt-bar-text">{duration_days}d</span>
                </div>
            </div>
        </div>
        """

    # Resource Load Distribution calculations
    resource_hours = {}
    for t in project_plan.tasks:
        for r in t.required_resources:
            clean_res = r.strip()
            if clean_res:
                resource_hours[clean_res] = resource_hours.get(clean_res, 0.0) + t.estimated_time_hours

    max_res_hours = max(list(resource_hours.values()) + [8.0])
    resource_bars = ""
    for name, hours in sorted(resource_hours.items()):
        width_pct = (hours / max_res_hours) * 100
        status_class = "warning" if hours > 40.0 else "normal"
        resource_bars += f"""
        <div class="resource-row">
            <div class="resource-label">{name}</div>
            <div class="resource-track">
                <div class="resource-bar {status_class}" style="width: {width_pct}%;">
                    <span class="resource-val">{round(hours, 1)} hrs</span>
                </div>
            </div>
        </div>
        """

    # Risks Registry section
    risks_html = ""
    if hasattr(project_plan, 'risks') and project_plan.risks:
        for r in project_plan.risks:
            severity_class = r.severity.lower().strip()
            risks_html += f"""
            <div class="risk-item">
                <div class="risk-header">
                    <span class="risk-badge {severity_class}">{r.severity}</span>
                    <span class="risk-name">{r.risk_name}</span>
                </div>
                <div class="risk-mitigation">
                    <strong>Mitigation:</strong> {r.mitigation_strategy}
                </div>
            </div>
            """
    else:
        risks_html = "<p class='no-data-msg'>No significant project risks identified by allocator agent.</p>"

    html_template = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Project Blueprint Dashboard</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Outfit:wght@600;700&display=swap" rel="stylesheet">
    <style>
        :root {{
            --bg-color: #0B0F19;
            --card-bg: rgba(17, 24, 39, 0.75);
            --border-color: rgba(245, 158, 11, 0.15);
            --accent-color: #F59E0B;
            --text-primary: #F3F4F6;
            --text-secondary: #9CA3AF;
            --red-color: #EF4444;
            --green-color: #10B981;
        }}
        
        body {{
            background-color: var(--bg-color);
            color: var(--text-primary);
            font-family: 'Inter', sans-serif;
            margin: 0;
            padding: 40px 20px;
            display: flex;
            justify-content: center;
        }}
        
        .container {{
            max-width: 1000px;
            width: 100%;
        }}
        
        header {{
            margin-bottom: 40px;
            border-bottom: 1px solid var(--border-color);
            padding-bottom: 20px;
        }}
        
        h1 {{
            font-family: 'Outfit', sans-serif;
            font-size: 2.5rem;
            color: var(--accent-color);
            margin: 0 0 10px 0;
            letter-spacing: -0.025em;
        }}
        
        .subtitle {{
            color: var(--text-secondary);
            font-size: 1.1rem;
            margin: 0;
        }}
        
        .section-title {{
            font-family: 'Outfit', sans-serif;
            font-size: 1.5rem;
            color: var(--text-primary);
            margin: 0 0 20px 0;
            border-left: 4px solid var(--accent-color);
            padding-left: 12px;
        }}
        
        .card {{
            background: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 24px;
            margin-bottom: 40px;
            backdrop-filter: blur(8px);
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
            text-align: left;
        }}
        
        th {{
            color: var(--text-secondary);
            font-weight: 500;
            font-size: 0.9rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            padding: 12px 16px;
            border-bottom: 1px solid var(--border-color);
        }}
        
        td {{
            padding: 16px;
            border-bottom: 1px solid rgba(255, 255, 255, 0.05);
            font-size: 0.95rem;
        }}
        
        tr:hover td {{
            background: rgba(245, 158, 11, 0.02);
        }}
        
        .task-name {{
            font-weight: 500;
        }}
        
        .badge {{
            background: rgba(245, 158, 11, 0.1);
            color: var(--accent-color);
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 0.8rem;
            font-weight: 500;
            margin-right: 6px;
            display: inline-block;
            border: 1px solid rgba(245, 158, 11, 0.2);
        }}
        
        .milestones-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
            gap: 20px;
        }}
        
        .milestone-card {{
            background: rgba(255, 255, 255, 0.02);
            border: 1px solid var(--border-color);
            border-radius: 6px;
            padding: 20px;
            transition: transform 0.2s, border-color 0.2s;
        }}
        
        .milestone-card:hover {{
            transform: translateY(-2px);
            border-color: var(--accent-color);
        }}
        
        .milestone-header {{
            display: flex;
            align-items: center;
            margin-bottom: 12px;
        }}
        
        .milestone-number {{
            background: var(--accent-color);
            color: var(--bg-color);
            font-family: 'Outfit', sans-serif;
            font-weight: 700;
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 0.85rem;
            margin-right: 12px;
        }}
        
        .milestone-title {{
            font-weight: 600;
            font-size: 1.1rem;
        }}
        
        .milestone-body p {{
            color: var(--text-secondary);
            font-size: 0.9rem;
            margin: 0;
            line-height: 1.5;
        }}

        /* Gantt Styles */
        .gantt-chart {{
            display: flex;
            flex-direction: column;
            gap: 16px;
            margin-top: 10px;
        }}

        .gantt-row {{
            display: flex;
            align-items: center;
        }}

        .gantt-label {{
            width: 200px;
            min-width: 200px;
            display: flex;
            gap: 10px;
            align-items: center;
            font-size: 0.9rem;
        }}

        .task-id {{
            background: rgba(255, 255, 255, 0.08);
            padding: 2px 6px;
            border-radius: 4px;
            font-weight: 600;
            font-size: 0.8rem;
        }}

        .task-title {{
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
            color: var(--text-primary);
        }}

        .gantt-track {{
            flex-grow: 1;
            background: rgba(255, 255, 255, 0.03);
            height: 28px;
            border-radius: 6px;
            position: relative;
            border: 1px solid rgba(255, 255, 255, 0.05);
        }}

        .gantt-bar {{
            position: absolute;
            height: 100%;
            background: linear-gradient(90deg, var(--accent-color), #D97706);
            border-radius: 5px;
            display: flex;
            align-items: center;
            justify-content: flex-end;
            padding-right: 8px;
            box-sizing: border-box;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);
        }}

        .gantt-bar-text {{
            font-size: 0.75rem;
            font-weight: 700;
            color: var(--bg-color);
        }}

        /* Resource Load Styles */
        .resource-chart {{
            display: flex;
            flex-direction: column;
            gap: 16px;
        }}

        .resource-row {{
            display: flex;
            align-items: center;
        }}

        .resource-label {{
            width: 150px;
            min-width: 150px;
            font-size: 0.9rem;
            font-weight: 500;
        }}

        .resource-track {{
            flex-grow: 1;
            background: rgba(255, 255, 255, 0.03);
            height: 24px;
            border-radius: 4px;
            border: 1px solid rgba(255, 255, 255, 0.05);
        }}

        .resource-bar {{
            height: 100%;
            background: var(--green-color);
            border-radius: 3px;
            display: flex;
            align-items: center;
            padding-left: 10px;
            box-sizing: border-box;
        }}

        .resource-bar.warning {{
            background: var(--red-color);
        }}

        .resource-val {{
            font-size: 0.75rem;
            font-weight: 700;
            color: var(--bg-color);
        }}

        /* Risks registry styles */
        .risks-container {{
            display: flex;
            flex-direction: column;
            gap: 16px;
        }}

        .risk-item {{
            background: rgba(255, 255, 255, 0.02);
            border: 1px solid var(--border-color);
            border-radius: 6px;
            padding: 16px;
        }}

        .risk-header {{
            display: flex;
            align-items: center;
            gap: 12px;
            margin-bottom: 8px;
        }}

        .risk-badge {{
            font-family: 'Outfit', sans-serif;
            font-weight: 700;
            font-size: 0.75rem;
            padding: 2px 8px;
            border-radius: 4px;
            text-transform: uppercase;
        }}

        .risk-badge.high {{
            background: rgba(239, 68, 68, 0.15);
            color: var(--red-color);
            border: 1px solid rgba(239, 68, 68, 0.3);
        }}

        .risk-badge.medium {{
            background: rgba(245, 158, 11, 0.15);
            color: var(--accent-color);
            border: 1px solid rgba(245, 158, 11, 0.3);
        }}

        .risk-badge.low {{
            background: rgba(16, 185, 129, 0.15);
            color: var(--green-color);
            border: 1px solid rgba(16, 185, 129, 0.3);
        }}

        .risk-name {{
            font-weight: 600;
            font-size: 1.05rem;
        }}

        .risk-mitigation {{
            font-size: 0.9rem;
            color: var(--text-secondary);
            line-height: 1.5;
        }}

        .no-data-msg {{
            color: var(--text-secondary);
            font-style: italic;
            margin: 0;
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>PROJECT BLUEPRINT</h1>
            <p class="subtitle">Generated dynamically by Multi-Agent Planning Suite</p>
        </header>
        
        <div class="card">
            <h2 class="section-title">Timeline (Gantt Chart)</h2>
            <div class="gantt-chart">
                {gantt_rows}
            </div>
        </div>

        <div class="card">
            <h2 class="section-title">Resource Utilization Analysis</h2>
            <div class="resource-chart">
                {resource_bars}
            </div>
        </div>

        <div class="card">
            <h2 class="section-title">Risk Assessment Registry</h2>
            <div class="risks-container">
                {risks_html}
            </div>
        </div>

        <div class="card">
            <h2 class="section-title">Execution Tasks Details</h2>
            <table>
                <thead>
                    <tr>
                        <th>ID</th>
                        <th>Task Name</th>
                        <th>Estimate</th>
                        <th>Allocations</th>
                    </tr>
                </thead>
                <tbody>
                    {tasks_html}
                </tbody>
            </table>
        </div>
        
        <div class="card">
            <h2 class="section-title">Milestones</h2>
            <div class="milestones-grid">
                {milestones_html}
            </div>
        </div>
    </div>
</body>
</html>
"""
    return html_template

# Chainlit setup
@cl.on_chat_start
async def start():
    # Initialize session variables to store inputs
    cl.user_session.set("inputs", {})
    cl.user_session.set("step", "llm_provider")
    await cl.Message(content="Welcome to the Project Planning Assistant! Let's start.\n\nWhich LLM Provider do you want to use? (Options: OpenAI, Groq, Ollama)").send()

@cl.on_message
async def main(message: cl.Message):
    current_step = cl.user_session.get("step")
    inputs = cl.user_session.get("inputs")

    # Store the user's response for the current step
    user_response = message.content.strip()
    if not user_response:
        await cl.Message(content="Please provide a valid input.").send()
        return

    # Process the input based on the current step
    if current_step == "llm_provider":
        provider_choice = user_response.lower()
        if provider_choice not in ["openai", "groq", "ollama"]:
            await cl.Message(content="Invalid choice. Please choose: OpenAI, Groq, or Ollama").send()
            return
        cl.user_session.set("llm_provider", provider_choice)
        cl.user_session.set("step", "ask_knowledge")
        await cl.Message(content="Would you like to upload any past project documents (TXT or PDF) to serve as a reference Knowledge Base? (Options: Yes, No)").send()

    elif current_step == "ask_knowledge":
        if user_response.lower() in ["yes", "y"]:
            files = await cl.AskFileMessage(
                content="Please upload a reference TXT or PDF file:",
                accept=["text/plain", "application/pdf"],
                max_size_mb=10,
                timeout=180
            ).send()
            if files:
                cl.user_session.set("knowledge_file_path", files[0].path)
                await cl.Message(content=f"Uploaded `{files[0].name}` successfully as reference Knowledge Base!").send()
        
        cl.user_session.set("step", "project_type")
        await cl.Message(content="Great! What is the project type? (e.g., Website, Mobile App)").send()

    elif current_step == "project_type":
        inputs["project_type"] = user_response
        cl.user_session.set("step", "project_objectives")
        await cl.Message(content="Great! What are the project objectives? (e.g., 'Create a website for a small business')").send()

    elif current_step == "project_objectives":
        inputs["project_objectives"] = user_response
        cl.user_session.set("step", "industry")
        await cl.Message(content="Nice! Which industry does this project belong to? (e.g., Technology, Healthcare)").send()

    elif current_step == "industry":
        inputs["industry"] = user_response
        cl.user_session.set("step", "team_members")
        await cl.Message(content="Got it! Please list the team members (e.g., 'Alice Smith (Project Manager), Bob Jones (Designer)')").send()

    elif current_step == "team_members":
        inputs["team_members"] = user_response
        cl.user_session.set("step", "project_requirements")
        await cl.Message(content="Almost there! What are the project requirements? (Provide a detailed list, one per line or as a paragraph)").send()

    elif current_step == "project_requirements":
        inputs["project_requirements"] = user_response
        cl.user_session.set("step", "task_approval")

        # All inputs collected, now process Phase 1 with CrewAI
        await cl.Message(content="Thank you! Processing Phase 1: Task Breakdown...").send()

        # Initialize LLM and Agents dynamically based on user's choice
        provider = cl.user_session.get("llm_provider", "openai")
        try:
            llm = get_llm(provider)
        except Exception as e:
            await cl.Message(content=f"Error initializing LLM for {provider}: {e}").send()
            return

        project_planning_agent = Agent(config=agents_config['project_planning_agent'], llm=llm)

        # Define task breakdown
        task_breakdown = Task(config=tasks_config['task_breakdown'], agent=project_planning_agent)

        # Get Knowledge Config
        file_path = cl.user_session.get("knowledge_file_path")
        knowledge_sources, embedder = get_knowledge_config(provider, file_path)

        # Create and run the crew for Phase 1
        crew = Crew(
            agents=[project_planning_agent],
            tasks=[task_breakdown],
            knowledge_sources=knowledge_sources,
            embedder=embedder,
            verbose=True
        )

        result = await cl.make_async(crew.kickoff)(inputs=inputs)
        task_list_str = result.raw
        cl.user_session.set("task_list_breakdown", task_list_str)

        response = (
            "### Phase 1: Proposed Task List Breakdown\n\n"
            f"{task_list_str}\n\n"
            "--- \n"
            "**Approval Required**:\n"
            "- Type **'approve'** to proceed with resource estimation and allocation.\n"
            "- Otherwise, describe any edits, additions, or refinements you would like to make."
        )
        await cl.Message(content=response).send()

    elif current_step == "task_approval":
        task_list_str = cl.user_session.get("task_list_breakdown")
        if user_response.lower() in ["approve", "yes", "y"]:
            cl.user_session.set("step", "done")
            await cl.Message(content="Task list approved! Proceeding to Phase 2: Estimation & Resource Allocation...").send()

            provider = cl.user_session.get("llm_provider", "openai")
            try:
                llm = get_llm(provider)
            except Exception as e:
                await cl.Message(content=f"Error initializing LLM for {provider}: {e}").send()
                return

            estimation_agent = Agent(
                config=agents_config['estimation_agent'],
                llm=llm,
                tools=[DuckDuckGoSearchRun()]
            )
            resource_allocation_agent = Agent(config=agents_config['resource_allocation_agent'], llm=llm)

            # Update project_requirements in inputs to be the approved task list
            inputs['project_requirements'] = task_list_str

            time_resource_estimation = Task(config=tasks_config['time_resource_estimation'], agent=estimation_agent)
            resource_allocation = Task(config=tasks_config['resource_allocation'], agent=resource_allocation_agent, output_pydantic=ProjectPlan)

            # Get Knowledge Config
            file_path = cl.user_session.get("knowledge_file_path")
            knowledge_sources, embedder = get_knowledge_config(provider, file_path)

            crew = Crew(
                agents=[estimation_agent, resource_allocation_agent],
                tasks=[time_resource_estimation, resource_allocation],
                knowledge_sources=knowledge_sources,
                embedder=embedder,
                verbose=True
            )

            result = await cl.make_async(crew.kickoff)(inputs=inputs)

            # Handle the result
            if hasattr(result, 'pydantic') and result.pydantic:
                project_plan = result.pydantic
            elif isinstance(result, ProjectPlan):
                project_plan = result
            else:
                try:
                    raw_data = json.loads(result.raw if hasattr(result, 'raw') else str(result))
                    project_plan = ProjectPlan(**raw_data)
                except (json.JSONDecodeError, AttributeError) as e:
                    raw_val = result.raw if hasattr(result, 'raw') else str(result)
                    await cl.Message(content=f"Error parsing result: {e}\nRaw output: {raw_val}").send()
                    return

            # Format and send the response
            tasks_df = pd.DataFrame(project_plan.model_dump()['tasks'])
            milestones_df = pd.DataFrame(project_plan.model_dump()['milestones'])

            response = (
                "## Final Project Plan\n\n"
                "### Tasks\n" + tasks_df.to_markdown(index=False) + "\n\n"
                "### Milestones\n" + milestones_df.to_markdown(index=False)
            )
            await cl.Message(content=response).send()

            # Save outputs (optional)
            with open('project_plan.json', 'w', encoding='utf-8') as f:
                json.dump(project_plan.model_dump(), f, indent=4)
            with open('Project_Planning.md', 'w', encoding='utf-8') as f:
                f.write(response)
            # Generate and save blueprint HTML dashboard
            html_dashboard = generate_dashboard_html(project_plan)
            with open('project_dashboard.html', 'w', encoding='utf-8') as f:
                f.write(html_dashboard)

            # Reset for a new project (optional)
            cl.user_session.set("inputs", {})
            cl.user_session.set("knowledge_file_path", None)
            cl.user_session.set("step", "llm_provider")
            await cl.Message(content="Project plan generated! Would you like to start a new project? If so, which LLM Provider do you want to use? (Options: OpenAI, Groq, Ollama)").send()
        else:
            # User provided refinement feedback
            await cl.Message(content="Refining the task list based on your feedback...").send()

            provider = cl.user_session.get("llm_provider", "openai")
            try:
                llm = get_llm(provider)
            except Exception as e:
                await cl.Message(content=f"Error initializing LLM for {provider}: {e}").send()
                return

            project_planning_agent = Agent(config=agents_config['project_planning_agent'], llm=llm)

            refine_task = Task(
                description=(
                    "You are a Project Planner. Refine the existing project task breakdown "
                    "based on the user's feedback. Make sure all original constraints and requirements "
                    "are still respected.\n\n"
                    "Existing Task Breakdown:\n{existing_tasks}\n\n"
                    "User Feedback / Edits:\n{feedback}"
                ),
                expected_output="An updated comprehensive list of tasks with detailed descriptions, timelines, and dependencies.",
                agent=project_planning_agent
            )

            # Get Knowledge Config
            file_path = cl.user_session.get("knowledge_file_path")
            knowledge_sources, embedder = get_knowledge_config(provider, file_path)

            crew = Crew(
                agents=[project_planning_agent],
                tasks=[refine_task],
                knowledge_sources=knowledge_sources,
                embedder=embedder,
                verbose=True
            )

            result = await cl.make_async(crew.kickoff)(inputs={"existing_tasks": task_list_str, "feedback": user_response})
            task_list_str = result.raw
            cl.user_session.set("task_list_breakdown", task_list_str)

            response = (
                "### Refined Task List Breakdown\n\n"
                f"{task_list_str}\n\n"
                "--- \n"
                "**Approval Required**:\n"
                "- Type **'approve'** to proceed with resource estimation and allocation.\n"
                "- Otherwise, describe any edits, additions, or refinements you would like to make."
            )
            await cl.Message(content=response).send()

if __name__ == "__main__":
    pass