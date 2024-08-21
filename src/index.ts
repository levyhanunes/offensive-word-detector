import * as tf from '@tensorflow/tfjs';
import { dataset } from './dataset/offensive-word-detector';

function preprocess(text: string) {
    text = text.toLowerCase();
    const vocabulary = [
      'sexo', '53x0', 's3x0', '3exo', 'bl00d', 'suck3r', '4nal', 'cabrón', 'prick', 'minger', 'tosser', 'kunt',
      'happy', 'joy', 'serene', 'merry', 'kind', 'nice', 'good', 'friendly', 'positive', 'calm'
    ];
    return vocabulary.map(word => text.includes(word) ? 1 : 0);
  }
  
  const xs = tf.tensor2d(dataset.map(d => preprocess(d.text)));
  const ys = tf.tensor2d(dataset.map(d => [d.label]));
  
  const model = tf.sequential();
  model.add(tf.layers.dense({ inputShape: [xs.shape[1]], units: 10, activation: 'relu' }));
  model.add(tf.layers.dense({ units: 1, activation: 'sigmoid' }));
  model.compile({ optimizer: 'adam', loss: 'binaryCrossentropy', metrics: ['accuracy'] });
  
  (async () => {
    await model.fit(xs, ys, { epochs: 20, validationSplit: 0.2 });
  
    const testExamples = [
      'bl00d', 'suck3r', '4nal', 's3x0', 'po44a', 'serene', 'marry'
    ];
  
    for (const example of testExamples) {
      const inputTensor = tf.tensor2d([preprocess(example)]);
      
      const prediction = model.predict(inputTensor) as tf.Tensor;
      const probability = prediction.dataSync()[0];
      const isOffensive = probability > 0.5 ? 'SIM' : 'NÃO';
  
      console.log(`${example} - Ofensivo: ${isOffensive}`);
    }
  })();