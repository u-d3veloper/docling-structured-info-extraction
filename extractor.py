from docling.document_converter import (
    DocumentConverter,
    ImageFormatOption,
    PdfFormatOption,
)
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableStructureOptions
from docling.datamodel.base_models import InputFormat
from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.datamodel.document import DocumentStream
import multiprocessing
import io


class Extractor:
    def __init__(self):
        cpu_cores = max(1, multiprocessing.cpu_count() - 1)
        accelerator_options = AcceleratorOptions(
            num_threads=cpu_cores,
            device=AcceleratorDevice.CPU,
        )

        pipeline_options = PdfPipelineOptions()
        pipeline_options.accelerator_options = accelerator_options
        pipeline_options.do_ocr = True
        pipeline_options.do_table_structure = True
        pipeline_options.table_structure_options = TableStructureOptions(
            do_cell_matching=True
        )

        self.converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
                InputFormat.IMAGE: ImageFormatOption(pipeline_options=pipeline_options),
            }
        )

    def extract_from_bytes(self, file_bytes: bytes, filename: str) -> str:
        """
        Extract text from a document represented as bytes.
        The bytes are wrapped in a BytesIO stream and passed to the DocumentConverter for processing.

        Args:
            file_bytes (bytes): The content of the document in bytes.
            filename (str): The name of the file, used for logging and error messages.

        Returns:
            str: The extracted text from the document in Markdown format.
        """
        byte_stream = io.BytesIO(file_bytes)
        doc_stream = DocumentStream(name=filename, stream=byte_stream)
        conversion_result = self.converter.convert(doc_stream)
        return conversion_result.document.export_to_markdown()
