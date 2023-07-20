import * as tf from '@tensorflow/tfjs';
import * as tfc from '@tensorflow/tfjs-converter';

window.tf = tf;
window.tfc = tfc;
window.progress = 0;
window.bytesUsed = 0;

tf.enableProdMode();

// Set model URL to load
const MODEL_URL = `${window.location.href}model_full/model.json`;

// Performing mirror padding
function mirrorPadFunc(input, pad_arr) {
  return tf.tidy(() => {
    for (let i = 0; i < 4; i++) {
      if (pad_arr[i][0] !== 0 || pad_arr[i][1] !== 0) {
        const slice_size = [-1, -1, -1, -1];
        slice_size[i] = pad_arr[i][0];
        const slice_begin = [0, 0, 0, 0];

        const padding_left = input.slice(slice_begin, slice_size);

        slice_size[i] = pad_arr[i][1];
        slice_begin[i] = input.shape[i] - pad_arr[i][1];

        const padding_right = input.slice(slice_begin, slice_size);

        input = tf.concat([padding_left, input, padding_right], i);
      }

      if (pad_arr[i][0] > 1 || pad_arr[i][1] > 1) {
        throw new Error(
          `Only input with no more than length one in padding is supported. We have: ${JSON.stringify(pad_arr)}`
        );
      }
    }
    return input;
  });
}

// For debugging purpose:
window.mirrorPadFunc = mirrorPadFunc;

// Array of progress values for debugging
const progressesList = [0.00023367749587460492, 0.054088046653978504, 0.1804816724673639, 0.18052037621199904, 0.2528568019649621, 0.37458444400475477, 0.39315031021211105, 0.39319017797911254, 0.4444196766347441, 0.5207431700988491, 0.550593651422125, 0.5542242372745627, 0.5605664132978859, 0.5806242652109398, 0.5927784050567816, 0.5962346785553008, 0.5981026434950807, 0.5989430676647844, 0.6435568450337933, 0.6676838282371483, 0.6684442258671517, 0.7463103400111626, 0.9019785470675509, 0.95];
let num_called = 0;

// Performing mirror padding asynchronously
const mirrorPad = async (node) => {
  // Calculate progress
  let progress = 0.9 * (performance.now() - start) / 15463.61999999499;

  /* progressesList.push(progress);
  console.log(progressesList); */

  // Get progress value from the array or set it to the maximum value if array length is exceeded
  if (num_called >= progressesList.length) {
    progress = 0.95;
  } else {
    progress = progressesList[num_called];
  }
  num_called += 1;

  // Update the global progress variable
  window.progress = progress;

  // Get memory information and update the global bytesUsed variable
  const memoryInfo = tf.memory();
  // console.log("Memory Info:", memoryInfo);
  window.bytesUsed = memoryInfo.numBytes;

  // Use normal pad (not mirror pad):
  // return tf.pad(node.inputs[0], node.inputs[1].arraySync(), 0);

  // Wait for the next animation frame
  await tf.nextFrame();

  // Check if the mode is not "reflect"
  if (node.attrs.mode !== "reflect") {
    throw new Error(`Only reflect mode is supported. Mode: ${node.attrs.mode}`);
  }

  // Get the pad_tensor and check if the input shape has a rank of 4
  const pad_tensor = node.inputs[1];
  if (node.inputs[0].shape.length === 4) {
    const pad_arr = await pad_tensor.array();
    const input = node.inputs[0];
    return mirrorPadFunc(input, pad_arr);
  } else {
    throw new Error(`Only input of rank 4 is supported. We have: ${JSON.stringify(pad_tensor.arraySync())}`);
  }
};

// Register the custom MirrorPad operation
tfc.registerOp('MirrorPad', mirrorPad);

// Generating images using the model
const generate = async (model, long_side_scale_size, img, output) => {
  console.log("Generation start");
  const img_tensor = tf.browser.fromPixels(img);
  let scaled_img_tensor;
  console.log("Original image size:", img_tensor.shape);

  // Scaling or Expansion based on the long_side_scale_size
  if (long_side_scale_size !== -1) {
    const scale_factor = Math.max(img_tensor.shape[0], img_tensor.shape[1]) / long_side_scale_size; // long side scaled size
    const scaled_size = [
      Math.round(img_tensor.shape[0] / scale_factor),
      Math.round(img_tensor.shape[1] / scale_factor)
    ];
    scaled_img_tensor = tf.tidy(() =>
      tf.image.resizeBilinear(img_tensor, scaled_size).expandDims(0).div(255)
    );
  } else {
    scaled_img_tensor = tf.tidy(() =>
      img_tensor.expandDims(0).div(255)
    );
  }

  start = performance.now();
  const generated = await model.executeAsync({ 'test': scaled_img_tensor });
  const end = performance.now();
  console.log("Image Generated");
  console.log(`Took ${(end - start) / 1000} s to generate the image`);

  tf.browser.toPixels((generated.squeeze(0).add(1)).div(2), output);
  generated.dispose();
  scaled_img_tensor.dispose();
};

// pre-heating the model
const preHeat = async () => {
  const model = await tfc.loadGraphModel(MODEL_URL);
  console.log("Model Loaded");
  model.dispose();
};

// Generating images based on the resize, fp16, img_id, and canvas_id
const generateImage = async (resize, fp16, img_id, canvas_id) => {
  tf.env().set('WEBGL_FORCE_F16_TEXTURES', fp16);

  let long_side_scale_size;

  switch (resize) {
    case "s":
      long_side_scale_size = 100;
      break;
    case "m":
      long_side_scale_size = 250;
      break;
    case "l":
      long_side_scale_size = 500;
      break;
    default:
      long_side_scale_size = -1;
  }

  const model = await tfc.loadGraphModel(MODEL_URL);
  console.log("Model Loaded");
  await generate(model, long_side_scale_size, document.getElementById(img_id), document.getElementById(canvas_id));
  tf.disposeVariables();
  console.log(tf.memory());
  window.progress = 1.0;
};

export { preHeat, generateImage };
