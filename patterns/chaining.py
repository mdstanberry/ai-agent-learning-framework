"""
Prompt Chaining Pattern

This module demonstrates the Prompt Chaining design pattern - a linear,
multi-step workflow where each step builds on the previous one.

Pattern Overview:
- Step 1: Generate an outline from a topic
- Gate: Validate the outline has minimum sections
- Step 2: Generate full content from the outline
- Step 3: Edit and refine the content

When to Use:
- When you have a predictable, linear sequence of steps
- When each step depends on the previous step's output
- When you want to break down a complex task into manageable parts
- When you need validation gates between steps

When NOT to Use:
- When the steps are unpredictable or dynamic
- When steps can happen in parallel
- When you need the agent to decide what to do next (use ReAct pattern instead)
"""

from typing import List, Optional
from pydantic import BaseModel, Field, field_validator
from utils.llm import LLMClient, call_llm
from utils.agent_logging import logger
from utils.schemas import ChainStep
from utils.config import config


# =============================================================================
# Pydantic Models for Chain Steps
# =============================================================================

class OutlineSection(BaseModel):
    """Represents a single section in an outline."""
    title: str = Field(description="Section title")
    description: str = Field(description="Brief description of what this section covers")
    key_points: List[str] = Field(description="Key points to cover in this section")


class Outline(BaseModel):
    """Complete outline structure."""
    topic: str = Field(description="The main topic")
    introduction: str = Field(description="Introduction paragraph summary")
    sections: List[OutlineSection] = Field(description="List of main sections")
    conclusion: str = Field(description="Conclusion paragraph summary")
    
    @field_validator('sections')
    @classmethod
    def validate_sections(cls, v: List[OutlineSection]) -> List[OutlineSection]:
        """Ensure we have at least some sections."""
        if len(v) < 1:
            raise ValueError("Outline must have at least one section")
        return v


class BlogPost(BaseModel):
    """Complete blog post content."""
    title: str = Field(description="Blog post title")
    introduction: str = Field(description="Introduction paragraph")
    sections: List[dict] = Field(description="List of sections, each with title and content")
    conclusion: str = Field(description="Conclusion paragraph")
    word_count: int = Field(description="Approximate word count")


class EditedPost(BaseModel):
    """Refined and edited blog post."""
    title: str = Field(description="Final blog post title")
    content: str = Field(description="Complete blog post content")
    improvements_made: List[str] = Field(description="List of improvements made during editing")
    final_word_count: int = Field(description="Final word count")


# =============================================================================
# Prompt Chaining Implementation
# =============================================================================

class PromptChain:
    """
    Implements the Prompt Chaining pattern for blog post generation.
    
    This class orchestrates a linear workflow:
    1. Generate outline from topic
    2. Validate outline meets requirements
    3. Generate full blog post from outline
    4. Edit and refine the blog post
    
    Each step uses the output of the previous step as input.
    """
    
    def __init__(self, provider: Optional[str] = None):
        """
        Initialize the prompt chain.
        
        Args:
            provider: LLM provider to use (defaults to config)
        """
        self.client = LLMClient(provider=provider)
        self.steps: List[ChainStep] = []
        self.min_sections = config.get_nested("patterns.chaining.min_outline_sections", 3)
        self.enable_validation = config.get_nested("patterns.chaining.enable_validation_gates", True)
    
    def generate_blog_post(self, topic: str) -> EditedPost:
        """
        Generate a complete blog post using the chaining pattern.
        
        This is the main entry point that orchestrates all steps.
        
        Args:
            topic: The topic to write about
            
        Returns:
            Final edited blog post
            
        Raises:
            ValueError: If validation gates fail
        """
        logger.section("Prompt Chaining: Blog Post Generation", f"Topic: {topic}")
        
        # Step 1: Generate Outline
        logger.info("Step 1: Generating outline...")
        outline = self._step1_generate_outline(topic)
        
        # Gate: Validate Outline
        if self.enable_validation:
            logger.info("Validating outline...")
            self._validate_outline(outline)
        
        # Step 2: Generate Full Post
        logger.info("Step 2: Generating full blog post from outline...")
        blog_post = self._step2_generate_post(outline)
        
        # Step 3: Edit and Refine
        logger.info("Step 3: Editing and refining blog post...")
        edited_post = self._step3_edit_post(blog_post)
        
        logger.success("Blog post generation complete!")
        return edited_post
    
    def _step1_generate_outline(self, topic: str) -> Outline:
        """
        Step 1: Generate an outline from the topic.
        
        Args:
            topic: The topic to create an outline for
            
        Returns:
            Structured outline
        """
        system_prompt = """You are an expert content strategist. 
Create a detailed outline for a blog post on the given topic.
The outline should be well-structured with clear sections."""
        
        user_prompt = f"""Create a detailed outline for a blog post about: {topic}

The outline should include:
- An introduction paragraph summary
- At least {self.min_sections} main sections, each with:
  - A clear title
  - A brief description
  - 3-5 key points to cover
- A conclusion paragraph summary

Make sure the outline is comprehensive and covers the topic thoroughly."""
        
        logger.thought(f"Creating outline for topic: {topic}")
        
        try:
            outline = self.client.call(
                messages=[{"role": "user", "content": user_prompt}],
                system_prompt=system_prompt,
                response_model=Outline
            )
            
            logger.success(f"Outline generated with {len(outline.sections)} sections")
            logger.info(f"Section titles: {[s.title for s in outline.sections]}")
            
            # Record step
            self.steps.append(ChainStep(
                step_number=1,
                step_name="Generate Outline",
                input=topic,
                output=outline.model_dump_json(),
                model_used=self.client.model_config.get("default_model")
            ))
            
            return outline
            
        except Exception as e:
            logger.error(f"Failed to generate outline: {e}", exception=e)
            raise
    
    def _validate_outline(self, outline: Outline) -> None:
        """
        Validation gate: Check if outline meets minimum requirements.
        
        Args:
            outline: The outline to validate
            
        Raises:
            ValueError: If validation fails
        """
        issues = []
        
        # Check minimum sections
        if len(outline.sections) < self.min_sections:
            issues.append(
                f"Outline has {len(outline.sections)} sections, "
                f"but minimum is {self.min_sections}"
            )
        
        # Check each section has content
        for i, section in enumerate(outline.sections):
            if not section.title.strip():
                issues.append(f"Section {i+1} has no title")
            if len(section.key_points) < 2:
                issues.append(f"Section '{section.title}' has fewer than 2 key points")
        
        if issues:
            error_msg = "Outline validation failed:\n" + "\n".join(f"  - {issue}" for issue in issues)
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        logger.success("Outline validation passed!")
    
    def _step2_generate_post(self, outline: Outline) -> BlogPost:
        """
        Step 2: Generate full blog post from the outline.
        
        Args:
            outline: The validated outline
            
        Returns:
            Complete blog post
        """
        system_prompt = """You are an expert blog writer. 
Write a comprehensive, engaging blog post based on the provided outline.
Make sure to cover all points mentioned in the outline."""
        
        # Format outline for the prompt
        outline_text = f"""
Topic: {outline.topic}

Introduction: {outline.introduction}

Sections:
"""
        for i, section in enumerate(outline.sections, 1):
            outline_text += f"""
{i}. {section.title}
   Description: {section.description}
   Key Points:
"""
            for point in section.key_points:
                outline_text += f"   - {point}\n"
        
        outline_text += f"\nConclusion: {outline.conclusion}"
        
        user_prompt = f"""Write a complete blog post based on this outline:

{outline_text}

Requirements:
- Write engaging, informative content
- Cover all key points from each section
- Use clear, readable language
- Aim for approximately 800-1200 words total
- Include a compelling title"""
        
        logger.thought("Expanding outline into full blog post content")
        
        try:
            blog_post = self.client.call(
                messages=[{"role": "user", "content": user_prompt}],
                system_prompt=system_prompt,
                response_model=BlogPost
            )
            
            logger.success(f"Blog post generated: {blog_post.word_count} words")
            logger.info(f"Title: {blog_post.title}")
            
            # Record step
            self.steps.append(ChainStep(
                step_number=2,
                step_name="Generate Blog Post",
                input=outline.model_dump_json(),
                output=blog_post.model_dump_json(),
                model_used=self.client.model_config.get("default_model")
            ))
            
            return blog_post
            
        except Exception as e:
            logger.error(f"Failed to generate blog post: {e}", exception=e)
            raise
    
    def _step3_edit_post(self, blog_post: BlogPost) -> EditedPost:
        """
        Step 3: Edit and refine the blog post.
        
        Args:
            blog_post: The blog post to edit
            
        Returns:
            Edited and refined blog post
        """
        system_prompt = """You are an expert editor. 
Review and improve the blog post for clarity, flow, and engagement.
Make specific improvements and note what you changed."""
        
        # Format blog post for editing
        post_content = f"""
Title: {blog_post.title}

Introduction:
{blog_post.introduction}

"""
        for section in blog_post.sections:
            post_content += f"""
{section.get('title', 'Section')}:
{section.get('content', '')}

"""
        
        post_content += f"""
Conclusion:
{blog_post.conclusion}

Word Count: {blog_post.word_count}
"""
        
        user_prompt = f"""Edit and improve this blog post:

{post_content}

Please:
- Improve clarity and readability
- Enhance flow between sections
- Fix any awkward phrasing
- Ensure consistency in tone
- Optimize for engagement
- Note all improvements made"""
        
        logger.thought("Editing and refining blog post for quality")
        
        try:
            edited_post = self.client.call(
                messages=[{"role": "user", "content": user_prompt}],
                system_prompt=system_prompt,
                response_model=EditedPost
            )
            
            logger.success(f"Blog post edited: {edited_post.final_word_count} words")
            logger.info(f"Improvements: {len(edited_post.improvements_made)} changes made")
            
            # Record step
            self.steps.append(ChainStep(
                step_number=3,
                step_name="Edit Blog Post",
                input=blog_post.model_dump_json(),
                output=edited_post.model_dump_json(),
                model_used=self.client.model_config.get("default_model")
            ))
            
            return edited_post
            
        except Exception as e:
            logger.error(f"Failed to edit blog post: {e}", exception=e)
            raise
    
    def get_steps(self) -> List[ChainStep]:
        """Get all chain steps for analysis."""
        return self.steps


if __name__ == "__main__":
    # Example usage
    print("=" * 60)
    print("Prompt Chaining Pattern Demo")
    print("=" * 60)
    
    # Create chain instance
    chain = PromptChain()
    
    # Generate a blog post
    topic = "Introduction to AI Agents"
    
    try:
        result = chain.generate_blog_post(topic)
        
        print("\n" + "=" * 60)
        print("Final Result")
        print("=" * 60)
        print(f"\nTitle: {result.title}")
        print(f"\nWord Count: {result.final_word_count}")
        print(f"\nImprovements Made:")
        for improvement in result.improvements_made:
            print(f"  - {improvement}")
        
        print("\n" + "=" * 60)
        print("Chain Steps Summary")
        print("=" * 60)
        for step in chain.get_steps():
            print(f"\nStep {step.step_number}: {step.step_name}")
            print(f"  Execution Time: {step.execution_time:.2f}s")
            if step.tokens_used:
                print(f"  Tokens Used: {step.tokens_used}")
        
    except Exception as e:
        print(f"\nError: {e}")
        print("\nNote: Make sure you have:")
        print("1. Copied .env.example to .env")
        print("2. Added your API key to .env")
        print("3. Installed all dependencies: pip install -r requirements.txt")



