import {
	IExecuteFunctions,
	INodeExecutionData,
	INodeType,
	INodeTypeDescription,
	NodeOperationError,
} from 'n8n-workflow';

import { Mistral } from '@mistralai/mistralai';

export class MistralOcr implements INodeType {
	description: INodeTypeDescription = {
		displayName: 'Mistral OCR',
		name: 'mistralOcr',
		icon: 'file:mistral.svg',
		group: ['transform'],
		version: 1,
		subtitle: '={{$parameter["operation"]}}',
		description: 'Extract text from PDF documents using Mistral OCR',
		defaults: {
			name: 'Mistral OCR',
		},
		inputs: ['main'],
		outputs: ['main'],
		credentials: [
			{
				name: 'mistralOcrApi',
				required: true,
			},
		],
		properties: [
			{
				displayName: 'Operation',
				name: 'operation',
				type: 'options',
				noDataExpression: true,
				options: [
					{
						name: 'Extract Text from PDF',
						value: 'extractText',
						description: 'Extract text from PDF document using Mistral OCR',
						action: 'Extract text from PDF',
					},
				],
				default: 'extractText',
			},
			{
				displayName: 'Input Type',
				name: 'inputType',
				type: 'options',
				options: [
					{
						name: 'Binary Data',
						value: 'binary',
						description: 'Use binary data from previous node',
					},
					{
						name: 'Base64 String',
						value: 'base64',
						description: 'Provide base64 encoded PDF string',
					},
					{
						name: 'URL',
						value: 'url',
						description: 'Provide URL to PDF document',
					},
				],
				default: 'binary',
				description: 'How to provide the PDF document',
			},
			{
				displayName: 'Binary Property',
				name: 'binaryPropertyName',
				type: 'string',
				default: 'data',
				required: true,
				displayOptions: {
					show: {
						inputType: ['binary'],
					},
				},
				description: 'Name of the binary property containing the PDF',
			},
			{
				displayName: 'Base64 Data',
				name: 'base64Data',
				type: 'string',
				default: '',
				required: true,
				displayOptions: {
					show: {
						inputType: ['base64'],
					},
				},
				description: 'Base64 encoded PDF data',
			},
			{
				displayName: 'Document URL',
				name: 'documentUrl',
				type: 'string',
				default: '',
				required: true,
				displayOptions: {
					show: {
						inputType: ['url'],
					},
				},
				description: 'URL to the PDF document',
			},
			{
				displayName: 'Model',
				name: 'model',
				type: 'string',
				default: 'mistral-ocr-latest',
				description: 'Mistral OCR model to use',
			},
			{
				displayName: 'Max Pages',
				name: 'maxPages',
				type: 'number',
				default: 0,
				description: 'Maximum number of pages to extract (0 = all pages)',
			},
			{
				displayName: 'Output Format',
				name: 'outputFormat',
				type: 'options',
				options: [
					{
						name: 'Markdown Array',
						value: 'markdownArray',
						description: 'Return array of markdown strings (one per page)',
					},
					{
						name: 'Combined Markdown',
						value: 'markdownCombined',
						description: 'Return single markdown string with all pages',
					},
					{
						name: 'JSON Object',
						value: 'json',
						description: 'Return JSON object with pages and metadata',
					},
				],
				default: 'json',
				description: 'How to format the output',
			},
			{
				displayName: 'Include Images',
				name: 'includeImages',
				type: 'boolean',
				default: false,
				description: 'Whether to include base64 encoded images in the response',
			},
		],
	};

	async execute(this: IExecuteFunctions): Promise<INodeExecutionData[][]> {
		const items = this.getInputData();
		const returnData: INodeExecutionData[] = [];

		const credentials = await this.getCredentials('mistralOcrApi');
		const apiKey = credentials.apiKey as string;

		const mistralClient = new Mistral({ apiKey });

		for (let i = 0; i < items.length; i++) {
			try {
				const operation = this.getNodeParameter('operation', i) as string;

				if (operation === 'extractText') {
					const inputType = this.getNodeParameter('inputType', i) as string;
					const model = this.getNodeParameter('model', i) as string;
					const maxPages = this.getNodeParameter('maxPages', i) as number;
					const outputFormat = this.getNodeParameter('outputFormat', i) as string;
					const includeImages = this.getNodeParameter('includeImages', i) as boolean;

					let documentUrl: string;

					// Prepare document based on input type
					if (inputType === 'binary') {
						const binaryPropertyName = this.getNodeParameter('binaryPropertyName', i) as string;
						const binaryData = this.helpers.assertBinaryData(i, binaryPropertyName);
						const buffer = await this.helpers.getBinaryDataBuffer(i, binaryPropertyName);
						const base64Data = buffer.toString('base64');
						documentUrl = `data:application/pdf;base64,${base64Data}`;
					} else if (inputType === 'base64') {
						const base64Data = this.getNodeParameter('base64Data', i) as string;
						documentUrl = `data:application/pdf;base64,${base64Data}`;
					} else {
						documentUrl = this.getNodeParameter('documentUrl', i) as string;
					}

					// Call Mistral OCR API
					const ocrResponse = await mistralClient.ocr.process({
						model,
						document: {
							type: 'document_url',
							document_url: documentUrl,
						},
						include_image_base64: includeImages,
					});

					if (!ocrResponse || !ocrResponse.pages) {
						throw new NodeOperationError(
							this.getNode(),
							'Mistral OCR returned invalid response',
							{ itemIndex: i }
						);
					}

					let pages = ocrResponse.pages.map((page: any) => page.markdown);
					const totalPages = ocrResponse.pages.length;

					// Apply max pages limit
					if (maxPages > 0) {
						pages = pages.slice(0, maxPages);
					}

					// Format output
					let json: any;
					if (outputFormat === 'markdownArray') {
						json = {
							pages,
							pageCount: totalPages,
							extractedPages: pages.length,
						};
					} else if (outputFormat === 'markdownCombined') {
						json = {
							markdown: pages.join('\n\n---\n\n'),
							pageCount: totalPages,
							extractedPages: pages.length,
						};
					} else {
						json = {
							pages,
							pageCount: totalPages,
							extractedPages: pages.length,
							model,
						};
					}

					returnData.push({
						json,
						pairedItem: i,
					});
				}
			} catch (error) {
				if (this.continueOnFail()) {
					returnData.push({
						json: {
							error: error.message,
						},
						pairedItem: i,
					});
					continue;
				}
				throw error;
			}
		}

		return [returnData];
	}
}
