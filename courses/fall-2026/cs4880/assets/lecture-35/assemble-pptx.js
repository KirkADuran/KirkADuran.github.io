// Export rendered Beamer pages as slides. Edit the LaTeX source to change content.
const fs = require('node:fs');
const path = require('node:path');
const PptxGenJS = require('pptxgenjs');

async function main() {
  const [renderedDir, outputFile] = process.argv.slice(2);
  if (!renderedDir || !outputFile) {
    throw new Error('Usage: node assemble-pptx.js <rendered-page-directory> <output.pptx>');
  }

  const pages = fs.readdirSync(renderedDir)
    .filter(name => /^page-\d+\.png$/.test(name))
    .sort((first, second) => Number(first.match(/\d+/)[0]) - Number(second.match(/\d+/)[0]));
  if (pages.length === 0) {
    throw new Error(`No page-*.png files found in ${renderedDir}`);
  }

  const pptx = new PptxGenJS();
  pptx.layout = 'LAYOUT_WIDE';
  pptx.author = 'CS 4880 Artificial Intelligence';
  pptx.subject = 'Transformers: From Tokens to Attention';
  pptx.title = 'CS4880 Lecture 35';
  pptx.company = 'CS 4880';
  pptx.lang = 'en-US';

  for (const [index, name] of pages.entries()) {
    const slide = pptx.addSlide();
    slide.addImage({ path: path.join(renderedDir, name), x: 0, y: 0, w: '100%', h: '100%' });
    slide.addNotes([
      '[Sources]',
      'Supplied slides: 25_480_W26.pdf and Copy of 24_480_W26.pdf.',
      'Original diagrams can be reproduced with generate-figures.py.',
      'Research references appear on the final slide.',
      `Slide ${index + 1}/${pages.length}.`,
    ].join('\n'));
  }

  await pptx.writeFile({ fileName: outputFile });
  console.log(`Wrote ${pages.length} slides to ${outputFile}`);
}

main().catch(error => {
  console.error(error.message);
  process.exitCode = 1;
});
