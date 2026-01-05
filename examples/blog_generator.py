"""
Blog Generator Example

This example demonstrates the Prompt Chaining pattern - a linear multi-step workflow
that generates a blog post through three sequential steps:
1. Generate an outline
2. Generate the full blog post
3. Edit and refine the post

Each step builds on the previous one, with validation gates between steps.

Run this example with:
    python examples/blog_generator.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from patterns.chaining import PromptChain
from utils.agent_logging import logger


def main():
    """
    Main function demonstrating the Prompt Chaining pattern.
    
    This workflow:
    1. Takes a topic as input
    2. Generates an outline (Step 1)
    3. Validates the outline has enough sections
    4. Generates a full blog post from the outline (Step 2)
    5. Edits and refines the blog post (Step 3)
    6. Returns the final edited post
    """
    print("=" * 70)
    print("Blog Generator - Prompt Chaining Pattern Demo")
    print("=" * 70)
    print("\nThis example demonstrates a 3-step workflow:")
    print("  1. Generate Outline")
    print("  2. Generate Blog Post")
    print("  3. Edit and Refine")
    print("\n" + "-" * 70)
    
    # Initialize the prompt chain
    chain = PromptChain()
    
    # Example topic
    topic = "Introduction to AI Agents"
    
    print(f"\n[Step 0] Input Topic: {topic}")
    print("-" * 70)
    
    try:
        # Step 1: Generate Outline
        print("\n[Step 1] Generating Outline...")
        print("This step creates a structured outline for the blog post.")
        outline_result = chain.generate_outline(topic)
        
        if outline_result.is_success:
            outline = outline_result.result
            print(f"\n[OK] Outline Generated Successfully!")
            print(f"Title: {outline.title}")
            print(f"Number of sections: {len(outline.sections)}")
            print("\nOutline sections:")
            for i, section in enumerate(outline.sections, 1):
                print(f"  {i}. {section.title}")
                if section.key_points:
                    for point in section.key_points[:2]:  # Show first 2 key points
                        print(f"     - {point}")
        else:
            print(f"\n[FAIL] Outline generation failed: {outline_result.error}")
            return
        
        # Validation gate: Check if outline has minimum sections
        print("\n[Validation Gate] Checking outline quality...")
        if len(outline.sections) < 3:
            print("[FAIL] Outline doesn't have enough sections. Minimum: 3")
            return
        print("[OK] Outline passes validation (has at least 3 sections)")
        
        # Step 2: Generate Blog Post
        print("\n" + "-" * 70)
        print("\n[Step 2] Generating Full Blog Post...")
        print("This step expands the outline into a complete blog post.")
        blog_result = chain.generate_blog_post(outline)
        
        if blog_result.is_success:
            blog_post = blog_result.result
            print(f"\n[OK] Blog Post Generated Successfully!")
            print(f"Title: {blog_post.title}")
            print(f"Word count: ~{len(blog_post.content.split())} words")
            print(f"\nFirst paragraph preview:")
            first_para = blog_post.content.split('\n\n')[0] if '\n\n' in blog_post.content else blog_post.content[:200]
            print(f"  {first_para[:200]}...")
        else:
            print(f"\n[FAIL] Blog post generation failed: {blog_result.error}")
            return
        
        # Step 3: Edit and Refine
        print("\n" + "-" * 70)
        print("\n[Step 3] Editing and Refining Blog Post...")
        print("This step improves clarity, flow, and overall quality.")
        edit_result = chain.edit_blog_post(blog_post)
        
        if edit_result.is_success:
            edited_post = edit_result.result
            print(f"\n[OK] Blog Post Edited Successfully!")
            print(f"Title: {edited_post.title}")
            print(f"Word count: ~{len(edited_post.content.split())} words")
            print(f"Improvements: {edited_post.improvements}")
            
            print("\n" + "=" * 70)
            print("FINAL RESULT")
            print("=" * 70)
            print(f"\nTitle: {edited_post.title}")
            print(f"\nContent Preview (first 500 characters):")
            print("-" * 70)
            print(edited_post.content[:500] + "...")
            print("-" * 70)
        else:
            print(f"\n[FAIL] Blog post editing failed: {edit_result.error}")
            return
        
        print("\n" + "=" * 70)
        print("SUCCESS: Blog post generated using Prompt Chaining pattern!")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n[ERROR] An error occurred: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

