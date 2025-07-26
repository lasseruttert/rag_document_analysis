# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **RAG (Retrieval-Augmented Generation) Document Analysis System** built in Python. The system processes multi-format documents (TXT/PDF/DOCX), creates semantic embeddings, stores them in a vector database, and provides intelligent question-answering capabilities through an advanced web interface with hybrid retrieval and comprehensive configuration management.


## Basic Commandments
- DONT be afraid to ask for clarification if the requirements are not clear.
- DONT make assumptions about the user's intent without sufficient context.
- DONT write code that is not necessary for the current task.
- DONT write code that is not relevant to the current project.
- DONT write code that is not maintainable or readable.
- USE meaningful, descriptive variable, function, and class names.
- USE consistent naming conventions (snake_case for variables and functions, CamelCase for classes).
- USE comments to explain complex logic, but avoid obvious comments.
- USE docstrings for all functions and classes to describe their purpose, parameters, and return values.
- USE type hints for function parameters and return types to improve code readability and maintainability.
- USE logging for debugging and information messages instead of print statements.
- NEVER simply agree with the user without understanding the context and requirements.
- NEVER use hardcoded values; always use constants or configuration files.
- ALWAYS write tests for new features and bug fixes to ensure code quality and functionality.
- ALWAYS follow the project's coding standards and guidelines.
- ALWAYS keep the codebase clean and organized, removing unused code and dependencies.
- ALWAYS document your code. 
- DO NOT update CLAUDE.md or GEMINI.md unless explicitly requested by the user.


## Environment Policies
- Only use the correct conda environment for this project: 'rag'
- Never install packages globally or in the base environment.
- Create and update a `requirements.txt` file after installing new packages.
- only run code in the 'rag' conda environment.


## Code Style
- Never use emojis in code comments or documentation.
- Always use English for code comments and documentation.
- Use `snake_case` for variable and function names.
- Use `CamelCase` for class names.
- Use `UPPER_CASE` for constants.
- Use `f-strings` for string formatting.
- Use `type hints` for function parameters and return types.
- Use `logging` for debugging and information messages instead of print statements.
- Use `docstrings` for all functions and classes to describe their purpose, parameters, and return values.


## Development Workflow Memories
- update PLAN.md and PROJECT.md after each successful implementation
- PLAN.md should contain a detailed plan for the next steps in the project, in a way that is easy to understand and follow, especially for a agent like Claude or Gemini.
- PROJECT.md should contain a high-level overview of the project, including its purpose, architecture, and key components.
- always run tests after making changes to ensure functionality


## Documentation
- Write documentation in Markdown format.
- Write documentation in a way that is easy to understand and follow, especially for a agent like Claude or Gemini.
- Write documentation in a way that reduces token usage for agents like Claude or Gemini.
- Use `README.md` for project overview, setup instructions, and usage examples.
- Use `CHANGELOG.md` for tracking changes, bug fixes, and new features.
- USE `PLAN.md` for detailed plans and next steps in the project.
- USE `FIX.md` for tracking issues, proposed fixes, and improvements.
- Always keep documentation up to date with code changes, especially after implementing new features or fixing bugs.


