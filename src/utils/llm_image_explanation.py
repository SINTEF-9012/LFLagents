"""
LLM Image Explanation Utilities for Manufacturing Data Visualization.

This module provides utilities for generating comprehensive explanations of
manufacturing data visualizations using Large Language Models with vision
capabilities, enabling multimodal analysis of charts and graphs.
"""

import base64
import os
from typing import Dict

from utils.llm_config import get_llm_instance


class LLMImageAnalyzer:
    """
    A class for analyzing manufacturing process images using LLM vision models.
    """

    # Constants
    DEFAULT_TEMPERATURE = 0.3  # Slightly higher for more creative analysis
    DEFAULT_MAX_TOKENS = 2000  # Increased for more detailed analysis

    # Shared factory context and instructions
    FACTORY_PROCESS_CONTEXT = """
    **FACTORY PROCESS CONTEXT:**
    The factory operates two main processes:

    **Storage Process:** User places raw goods manually into the Delivery and Pick up station. Then the vacuum gripper robot (VGR) moves to the storage position, picks up the raw good and moves it first to the colour sensor, and then to the NFC reader. The high-bay warehouse (HBW) provides an empty container to hold the raw goods. VGR picks it up using its peel and places it on the input/output device. The VGR's arm turns into the discharge position, moves down and places the raw goods in the provided container. The container with the raw goods is transported in the input/output device to the HBW gripper arm pick-up position. The container is picked up here, moved to the storage position in the HBW, and stored there.

    **Production Process:** The transport arm of the high-bay warehouse moves to the storage position for a ordered workpiece. Pick it up and place it in the input/output station of the high-bay warehouse. There, it is moved to the pick-up position of the vacuum gripper. The vacuum gripper picks up the workpiece at the pick-up position and places it on the peel of the furnace. The workpiece is pushed in, burned, and moved back out of the furnace. The provided transport sled with vacuum suction cup transports it to the "Milling" processing machine. After the milling process is complete, the workpiece will be turned horizontally by 90 degrees and pushed onto a conveyor belt using pneumatic equipment. The workpiece goes through colour detection on the way to the sorting device. Depending on the detected colour, the workpiece is pushed pneumatically from the conveyor belt in front of a material slide. The workpiece is in a pick-up area and can be transported from here using the vacuum gripper. The gripper places the workpiece on the NFC reader for final labelling. There, it is labelled with relevant workpiece data such as the order date, manufacturing or delivery date. Then the workpiece is placed in the output compartment of the input/output station.

    **KEY COMPONENTS:**
    - VGR: Vacuum Gripper Robot (handles workpiece movement and positioning)
    - HBW: High-Bay Warehouse (storage and retrieval system)
    - SLD: Sorting Line Device (conveyor belt and sorting mechanism)
    - MPO: Multi-Processing Oven (furnace for processing workpieces)
    - DPS: Delivery and Pick-up Station (input/output interface)
    """

    CHART_READING_INSTRUCTIONS = """
    **CHART READING INSTRUCTIONS:**
    - Y-axis shows position values (0-5000+ range)
    - X-axis shows time progression
    - Different colored lines represent different components
    - Status annotations show component states at specific times
    - Look for coordinated movements between components
    - Identify start/end times of distinct operational phases
    - Analyze position trajectories for movement patterns and cycle identification
    """

    def __init__(
        self,
        model: str = None,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ):
        """
        Initialize the LLM image analyzer.

        Args:
            model: The LLM model to use (if None, uses CHAT_MODEL from env)
            temperature: Temperature for response generation
            max_tokens: Maximum tokens in response
        """
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.llm = get_llm_instance(
            model_name=model, temperature=temperature, max_tokens=max_tokens
        )

    def _read_image_as_base64(self, image_path: str) -> str:
        """
        Read and encode image as base64.

        Args:
            image_path: Path to the image file

        Returns:
            Base64 encoded image string

        Raises:
            FileNotFoundError: If image file doesn't exist
            IOError: If unable to read image
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Chart image not found at {image_path}")

        try:
            with open(image_path, "rb") as image_file:
                image_data = image_file.read()
            return base64.b64encode(image_data).decode()
        except IOError as e:
            raise IOError(f"Unable to read image file: {e}")

    def _create_vision_message(self, prompt: str, image_base64: str) -> Dict:
        """
        Create a vision message for the LLM.

        Args:
            prompt: The analysis prompt
            image_base64: Base64 encoded image

        Returns:
            Formatted message dictionary
        """
        return {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{image_base64}"},
                },
            ],
        }

    def _get_response_content(self, response) -> str:
        """
        Extract content from LLM response.

        Args:
            response: LLM response object

        Returns:
            Response content as string
        """
        return response.content if hasattr(response, "content") else str(response)

    def _create_data_enhanced_prompt(self, process_data_summary: str) -> str:
        """
        Create an analysis prompt that combines chart image with actual process data.

        Args:
            process_data_summary: Summary of actual process data and metrics

        Returns:
            Data-enhanced prompt string
        """
        data_analysis_requirements = """
        **CRITICAL ANALYSIS REQUIREMENTS:**
        1. Base your analysis on BOTH the chart visualization AND the provided process data
        2. Reference specific timestamps, position values, and status codes from the chart
        3. Correlate chart patterns with the quantitative metrics in the data summary
        4. Identify distinct operational cycles visible in both sources
        5. Use exact data points - avoid generic statements
        6. Automatically detect whether this shows Storage Process, Production Process, or both
        """

        data_analysis_structure = """
        #### Process Type Identification
        Based on the specific chart patterns and the provided data analysis, automatically identify whether this shows storage process, production process, or both. Reference the cycle count and component activities from both sources and provide evidence-based reasoning.

        #### Technical Metrics & Cycle Analysis
        Provide comprehensive technical analysis using both chart and data:
        - Reference the exact number of cycles from the data summary and correlate with visual patterns in the chart
        - Use the specific cycle durations and timing intervals provided to identify process phases
        - Correlate chart position movements with the quantitative ranges and analyze movement efficiency
        - Reference the exact component utilization percentages and identify optimization opportunities
        - Analyze status transitions and their timing relative to position movements
        - Use the specific efficiency metrics to assess process performance

        #### Process Sequence Analysis
        Describe the detailed operational sequence using integrated chart and data insights:
        - Map exact timestamps from the chart timeline to process steps
        - Use the cycle start/end times from the data summary to identify phase boundaries
        - Correlate chart position movements with component coordination patterns
        - Identify handoff points between components and measure handoff efficiency
        - Reference the component coordination patterns from both visual and quantitative sources

        **RESPONSE REQUIREMENTS:**
        - Use specific timestamps from the chart (e.g., "at 14:05:30")
        - Reference exact position values from the chart (e.g., "VGR moved from position 1000 to 3000")
        - Use the exact cycle count, durations, and intervals from the data summary
        - Reference the specific component utilization percentages provided
        - Correlate chart patterns with the quantitative metrics in detail
        - Provide intelligent insights that go beyond simple observation

        Start your response directly with "#### Process Type Identification" at level 4 header - do not include any introductory phrases. Write for factory operators who need clear, actionable information about the process flow.
        """

        return f"""Analyze this fischertechnik training factory time-series chart in conjunction with the actual process data provided below.

        {data_analysis_requirements}

        {self.CHART_READING_INSTRUCTIONS}

        {self.FACTORY_PROCESS_CONTEXT}

        **ACTUAL PROCESS DATA SUMMARY:**
        {process_data_summary}

        **ANALYSIS TASK:**
        Using both the chart visualization and the actual process data above, provide a comprehensive analysis that:
        1. Correlates specific visual patterns in the chart with the quantified process data
        2. References exact timestamps and position values from the chart
        3. Uses the cycle analysis data to identify distinct operational phases
        4. Compares observed chart patterns with the quantitative metrics provided
        5. Identifies component coordination patterns visible in both sources
        6. Automatically determines the process type based on the evidence

        **MANDATORY DATA USAGE:**
        - Reference the specific cycle count and timing from the data summary
        - Use the exact position ranges and movement patterns provided
        - Correlate chart timestamps with the data-driven insights
        - Reference the component utilization percentages and status transitions
        - Use the detected process type and efficiency metrics

        {data_analysis_structure}"""

    def analyze_with_data_and_image(
        self, image_path: str, process_data_summary: str
    ) -> str:
        """
        Generate comprehensive process explanation combining chart image and actual process data.

        Args:
            image_path: Path to the saved chart image (PNG format)
            process_data_summary: Summary of actual process data and metrics

        Returns:
            Comprehensive explanation combining visual and data analysis
        """
        # Read and encode the image
        image_base64 = self._read_image_as_base64(image_path)

        # Create data-enhanced prompt and message
        prompt = self._create_data_enhanced_prompt(process_data_summary)
        message = self._create_vision_message(prompt, image_base64)

        # Get response from vision model
        response = self.llm.invoke([message])
        return self._get_response_content(response)

    def analyze_with_data_image_and_custom_prompt(
        self, image_path: str, process_data_summary: str, custom_prompt: str
    ) -> str:
        """
        Generate comprehensive process explanation combining chart image, actual process data, and custom user prompt.

        Args:
            image_path: Path to the saved chart image (PNG format)
            process_data_summary: Summary of actual process data and metrics
            custom_prompt: Custom analysis request from the user

        Returns:
            Comprehensive explanation combining visual, data, and custom analysis
        """
        # Read and encode the image
        image_base64 = self._read_image_as_base64(image_path)

        # Create enhanced prompt that incorporates custom request and data
        base_prompt = self._create_data_enhanced_prompt(process_data_summary)
        enhanced_prompt = f"""{base_prompt}

        **ADDITIONAL CUSTOM ANALYSIS REQUEST:**
        {custom_prompt.strip()}

        Please address this specific custom request while maintaining the structured analysis format above. Use both the chart visualization and the provided process data to answer the custom request thoroughly."""

        # Create message with enhanced prompt
        message = self._create_vision_message(enhanced_prompt, image_base64)

        # Get response from vision model
        response = self.llm.invoke([message])
        return self._get_response_content(response)


class ChartImageSaver:
    """
    A class for saving Plotly charts as high-quality images.
    """

    # Constants
    DEFAULT_WIDTH = 1400
    DEFAULT_HEIGHT = 900
    DEFAULT_SCALE = 3
    DEFAULT_FORMAT = "png"
    DEFAULT_ENGINE = "kaleido"

    def __init__(
        self,
        width: int = DEFAULT_WIDTH,
        height: int = DEFAULT_HEIGHT,
        scale: int = DEFAULT_SCALE,
        format: str = DEFAULT_FORMAT,
        engine: str = DEFAULT_ENGINE,
    ):
        """
        Initialize the chart image saver.

        Args:
            width: Image width in pixels
            height: Image height in pixels
            scale: Scale factor for image quality
            format: Image format (png, jpg, etc.)
            engine: Engine to use for image export
        """
        self.width = width
        self.height = height
        self.scale = scale
        self.format = format
        self.engine = engine

    def _ensure_directory_exists(self, filepath: str) -> None:
        """
        Ensure the directory for the file exists.

        Args:
            filepath: Path to the file
        """
        directory = os.path.dirname(filepath)
        if directory:
            os.makedirs(directory, exist_ok=True)

    def _configure_figure_for_export(self, fig):
        """
        Configure figure settings for optimal PNG export.

        Args:
            fig: Plotly figure object

        Returns:
            Configured figure
        """
        fig.update_layout(
            template="plotly_white",  # Use white template for better PNG export
            plot_bgcolor="white",  # White background
            paper_bgcolor="white",  # White paper background
            font=dict(color="black"),  # Ensure text is black on white background
        )
        return fig

    def save_chart(self, fig, filepath: str) -> bool:
        """
        Save a plotly figure as a high-quality PNG image with preserved colors.

        Args:
            fig: Plotly figure object
            filepath: Path where to save the image

        Returns:
            True if successful, False otherwise
        """
        try:
            # Ensure the directory exists
            self._ensure_directory_exists(filepath)

            # Configure the figure for export (note: modifies the original)
            configured_fig = self._configure_figure_for_export(fig)

            # Export the image
            configured_fig.write_image(
                filepath,
                format=self.format,
                width=self.width,
                height=self.height,
                scale=self.scale,
                engine=self.engine,
            )

            # print(f"Successfully saved chart: {filepath}")
            return True

        except Exception as e:
            print(f"Error saving chart: {e}")
            return False


def save_chart_as_image(fig, filepath: str) -> bool:
    """
    Save a plotly figure as a high-quality PNG image with preserved colors.

    Args:
        fig: Plotly figure object
        filepath: Path where to save the image

    Returns:
        True if successful, False otherwise
    """
    saver = ChartImageSaver()
    return saver.save_chart(fig, filepath)


def generate_comprehensive_process_explanation(
    image_path: str,
    process_data_summary: str,
    custom_prompt: str = None
) -> str:
    """
    Generate comprehensive process explanation using data + image analysis for optimal results.

    This function provides the most comprehensive analysis by:
    - Always using data + image combination for detailed analysis
    - LLM automatically detects process type from data and chart
    - Provides intelligent insights beyond simple observation
    - Includes efficiency analysis, bottleneck identification, and optimization recommendations
    - Optionally incorporates custom user requirements for enhanced insights

    Args:
        image_path: Path to the saved chart image (PNG format)
        process_data_summary: Summary of actual process data and metrics (required)
        custom_prompt: Optional custom analysis request from the user

    Returns:
        Comprehensive explanation using data + image analysis with optional custom focus
    """
    # Use enhanced settings for more detailed and intelligent analysis
    analyzer = LLMImageAnalyzer(temperature=0.1, max_tokens=2000)

    # Always use comprehensive data + image analysis
    if custom_prompt and custom_prompt.strip():
        print("Generating analysis (data + image + custom prompt) with insights...")
        return analyzer.analyze_with_data_image_and_custom_prompt(
            image_path, process_data_summary, custom_prompt
        )
    else:
        print("Generating analysis (data + image) with insights...")
        return analyzer.analyze_with_data_and_image(image_path, process_data_summary)
