# Idea on how the workflow of this autonomous virtual ai lab might work
## Sequential domain specific agents
### Each agent has its own CLAUDE.md

1. Researcher (folder `./researcher` | SKILLS: brainstorm, hf-cli, search )
  1. RNG:
    * Brainstorm with full freedom and creativity.
    * Search online for strategies/methods/architectures/theories/papers/studies/projects.
    * hf-cli -> get daily papers.
    * Combos of above
  2. Evaluate and hypothesis by being creative but at the same time believing that the experiment will at least be able to train the full training cycle.
  3. Write an `HYPOTHESIS.md`, taking into consideration that the final model of the experiment should be less than 200M parameters, under an `<experiment-readable-code-name->` folder.
  4. Review it's own perfomance and write a `REVIEW.md` with a sequential and well documented review of its agentic session.

2. Planner (folder `./planner` | SKILLS: writing-plans, pytorch-patterns )
  1. From given `<experiment-readable-code-name->` folder, read `HYPOTHESIS.md`.
  2. Think of a plan on how to implement the pytorch pieces.
  3. Write the `PLAN.md` under the experiment's folder.
  4. Review the `PLAN.md` to check if it makes sense and if it is comprehensive for the next agent (Builder) to ensure it will correctly implement the experiment 
  5. Update the plan if necessary
  6. Review it's own perfomance and write a `REVIEW.md` with a sequential and well documented review of its agentic session.

3. Engineer (folder `./engineer` | SKILLS: executing-plans, pytorch-patterns )
  1. Study the given `PLAN.md` under the experiment's folder.
  2. Prepare a scaffold for the experiment's implementation.
  3. Build the necessary code (sub)modules while following the pytorch-patterns skill
  4. Run very small smoke tests to make sure the experiment will run on the full training cycle
  5. Fix/Debug if necessary
  6. Write a comprehensive `README.md` file to explain the code so that the next agent (Trainer) know how to use the (sub)modules to create the model and train it
  6. Review it's own perfomance and write a `REVIEW.md` with a sequential and well documented review of its agentic session.

4. Trainer (folder `./trainer` | SKILLS: pytorch-patterns)
  1. Reads the given `README.md` of the experiment to understand on how to implement the model on the training pipeline
  2. Configures the training pipeline based on the experiment's model and `README.md`
  3. Trains the model on a very small scale dataset (`datasets/babylm-strict-small/*.parquet`) for one epoch
  4. Evaluates the training on this very small scale dataset to check if the model is learning and if the training pipeline is properly optimized (training speed)
  5. Makes necessary changes if necessary
  6. Decide the training pipeline configuration for the full training cycle (scaling laws, epochs, lr, batch size, total batch size, etc...)
  6. Start the full training pipeline on the full dataset (`datasets/gpt-training-small/*.parquet`)
  7. Review it's own perfomance and write a `REVIEW.md` with a sequential and well documented review of its agentic session.
  8. Start a 15 minute timer to prompt the next agent (Reporter).
    
  
5. Reporter (folder `./reporter` | SKILLS: TBD... possibly 'llm-evaluation')
  1. Triggered after 15 minutes have passed since the training pipeline was executed.
  2. Checks the training status and triggers an interruption through a CLI
  3. The interruption will make the traning cycle to save a checkpoint on it's latest step
  4. Quickly evaluates the perfomance from the latest checkpoints and decides which one should be the best candidate (each checkpoint should have a json that summarizes the training perfomance on that checkpoint)
  5. Write a indepth report about the results of the top-3 candidates of saved checkpoints
  6. Deletes the checkpoints that were not chosen for the top-3
  7. Review it's own perfomance and write a `REVIEW.md` with a sequential and well documented review of its agentic session.

6. Reviewer (folder `./reviewer` | SKILLS: agent-evaluation, self-improving-agent)
  1. Checks the work of each agent and reviews their perfomance by reading each `REVIEW.md`
  2. Decides if a certain agent needs to have their `AGENTS.md` updated and/or create/update their SKILLS
  3. Create/Updates if necessary
  4. Gives a final review of the whole experiment including perfomance of agents, proposed and implemented improvements for each agent, the experiment perfomance (training speed and loss, model perfomance), observations, possible improvements, possible new ideas, conclusions.
    
7. Loop back to Researcher  

### SKILLS
- `npx skills add https://github.com/charon-fan/agent-playbook --skill self-improving-agent`
- `npx skills add https://github.com/obra/superpowers --skill executing-plans`
- `npx skills add https://github.com/obra/superpowers --skill writing-plans`
- `npx skills add https://github.com/obra/superpowers --skill brainstorming`
- `npx skills add https://github.com/tavily-ai/skills --skill search`
- `npx skills add https://github.com/affaan-m/everything-claude-code --skill pytorch-patterns`
- `npx skills add https://github.com/wshobson/agents --skill llm-evaluation`
- `npx skills add https://github.com/supercent-io/skills-template --skill agent-evaluation`
- `npx skills add https://github.com/huggingface/skills --skill hf-cli`
