import * as tf from "@tensorflow/tfjs";
import * as mobilenet from "@tensorflow-models/mobilenet";
import * as knnClassifier from "@tensorflow-models/knn-classifier";

export interface CustomPrediction {
  topLabel: string;
  topConfidence: number;
  confidences: Record<string, number>;
}

export interface SerializedCustomModel {
  version: number;
  savedAt: string;
  classes: string[];
  dataset: Record<string, { shape: number[]; data: number[] }>;
}

const STORAGE_KEY = "office_vision_custom_model_v1";

/**
 * In-App Transfer Learning AI Engine:
 * Uses MobileNet v2 deep feature embeddings + KNN Classifier so users can train
 * custom office activities/objects (Using Phone, Reading a Book, Working at PC,
 * or ANY custom class) directly on their live webcam in seconds.
 */
export class CustomActivityModel {
  private net: mobilenet.MobileNet | null = null;
  private classifier: knnClassifier.KNNClassifier = knnClassifier.create();

  async load(): Promise<void> {
    await tf.ready();
    this.net = await mobilenet.load({
      version: 2,
      alpha: 1.0,
      modelUrl: "/models/mobilenet_v2/model.json",
    });
  }

  isLoaded(): boolean {
    return this.net !== null;
  }

  getClassCounts(): Record<string, number> {
    return this.classifier.getClassExampleCount();
  }

  getTotalExamples(): number {
    return Object.values(this.classifier.getClassExampleCount()).reduce(
      (sum, n) => sum + n,
      0
    );
  }

  addExample(
    source: HTMLVideoElement | HTMLImageElement | HTMLCanvasElement,
    classLabel: string
  ): number {
    if (!this.net) {
      throw new Error("MobileNet feature extractor is still loading.");
    }
    const activation = this.net.infer(source, true) as tf.Tensor;
    this.classifier.addExample(activation, classLabel);
    activation.dispose();
    return this.classifier.getClassExampleCount()[classLabel] || 0;
  }

  async predict(
    source: HTMLVideoElement | HTMLImageElement | HTMLCanvasElement,
    k = 10
  ): Promise<CustomPrediction | null> {
    if (!this.net || this.classifier.getNumClasses() === 0) {
      return null;
    }
    const activation = this.net.infer(source, true) as tf.Tensor;
    try {
      const result = await this.classifier.predictClass(activation, k);
      const confidences: Record<string, number> = {};
      for (const [label, score] of Object.entries(result.confidences)) {
        confidences[label] = score;
      }
      return {
        topLabel: result.label,
        topConfidence: result.confidences[result.label] || 0,
        confidences,
      };
    } finally {
      activation.dispose();
    }
  }

  clearClass(classLabel: string): void {
    const counts = this.classifier.getClassExampleCount();
    if (counts[classLabel]) {
      this.classifier.clearClass(classLabel);
    }
  }

  clearAll(): void {
    this.classifier.clearAllClasses();
  }

  serialize(classes: string[]): SerializedCustomModel {
    const dataset = this.classifier.getClassifierDataset();
    const out: Record<string, { shape: number[]; data: number[] }> = {};
    for (const [label, tensor] of Object.entries(dataset)) {
      out[label] = {
        shape: tensor.shape,
        data: Array.from(tensor.dataSync()),
      };
    }
    return {
      version: 1,
      savedAt: new Date().toISOString(),
      classes,
      dataset: out,
    };
  }

  deserialize(serialized: SerializedCustomModel): string[] {
    this.classifier.clearAllClasses();
    const tensorMap: Record<string, tf.Tensor2D> = {};
    for (const [label, entry] of Object.entries(serialized.dataset || {})) {
      tensorMap[label] = tf.tensor2d(
        entry.data,
        entry.shape as [number, number]
      );
    }
    if (Object.keys(tensorMap).length > 0) {
      this.classifier.setClassifierDataset(tensorMap);
    }
    return serialized.classes || Object.keys(tensorMap);
  }

  saveToLocalStorage(classes: string[]): void {
    const payload = this.serialize(classes);
    localStorage.setItem(STORAGE_KEY, JSON.stringify(payload));
  }

  loadFromLocalStorage(): string[] | null {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return null;
    try {
      const parsed = JSON.parse(raw) as SerializedCustomModel;
      return this.deserialize(parsed);
    } catch {
      return null;
    }
  }
}
