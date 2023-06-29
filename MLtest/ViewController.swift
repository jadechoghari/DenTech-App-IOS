//
//  ViewController.swift
//  MLtest
//
//  Created by Jade Choghari on 28/06/2023.
//

import UIKit
import CoreML
import Vision

class ViewController: UIViewController, UIImagePickerControllerDelegate, UINavigationControllerDelegate {

    @IBOutlet weak var imageView: UIImageView!
    let imagePicker = UIImagePickerController()
    
    
    override func viewDidLoad() {
        super.viewDidLoad()
        // Do any additional setup after loading the view.
        
        imagePicker.delegate = self
        imagePicker.sourceType = .photoLibrary
        imagePicker.allowsEditing = false
    }
    
    func imagePickerController(_ picker: UIImagePickerController, didFinishPickingMediaWithInfo info: [UIImagePickerController.InfoKey : Any]) {
        
        if let userPickedImage = info[UIImagePickerController.InfoKey.originalImage] as? UIImage {
            imageView.image = userPickedImage
            
            guard let ciimage = CIImage(image: userPickedImage) else {
                fatalError("Could not convert CIImage")
            }
            detect(image: ciimage)
        }
        imagePicker.dismiss(animated: true, completion: nil)
    }
    
    let idxToClass: [Int: String] = [
        0: "amalgam tattoo",
        1: "aphthus ulcer",
        2: "candidiasis1",
        3: "fibroma",
        4: "geographic tongue",
        5: "hemangioma",
        6: "leukoplakia-hyperkeratosis",
        7: "lichen planus",
        8: "melanotic macule",
        9: "mucocele",
        10: "papilloma",
        11: "pyogenic granuloma",
        12: "squamous cell carcinoma",
        13: "tongue cheek chewing"
    ]

    func detect(image: CIImage) {
        // Initialize MLModelConfiguration
        let config = MLModelConfiguration()

        // Load the Core ML model
        do {
            let model = try EfficientNet_b3(configuration: config)

            // Resize and convert image to MLMultiArray
            let resizedImage = resize(image: image, newSize: CGSize(width: 256, height: 256))
            guard let pixelBuffer = convertImageToBuffer(resizedImage),
                  let multiArray = try? convertPixelBufferToMultiArray(pixelBuffer: pixelBuffer) else {
                print("Conversion to MLMultiArray failed.")
                return
            }

            // Create an input instance
            let input = EfficientNet_b3Input(x_1: multiArray)

            // Perform the prediction
            if let output = try? model.prediction(input: input) {
                // HandlING the output multi array according to  model's specifics <implement here>
                print("This is the output: ", output.var_1837)
                
                // Convert the MLMultiArray to a Swift Array
                let multiArray = output.var_1837
                let count = multiArray.count
                var array = [Float](repeating: 0, count: count)
                for i in 0..<count {
                    array[i] = multiArray[i].floatValue
                }

                // Compute softmax to get probabilities
                let exps = array.map { exp($0) }
                let sum = exps.reduce(0, +)
                let probabilities = exps.map { $0 / sum }
                
                // Get top 3 classes
                var classProbabilities = zip(idxToClass.indices, probabilities).sorted(by: { $0.1 > $1.1 })
                classProbabilities = Array(classProbabilities.prefix(3))

                // Extract the actual classes and probabilities
                let topClasses = classProbabilities.map { idxToClass[$0.0] }
                let topP = classProbabilities.map { $0.1 }

                print(topClasses, topP)
            }
 else {
                print("Model failed to process the image.")
            }
        } catch {
            print("Error loading CoreML model:", error)
        }
    }


    func resize(image: CIImage, newSize: CGSize) -> CIImage {
        let scaleX = newSize.width / image.extent.size.width
        let scaleY = newSize.height / image.extent.size.height
        return image.transformed(by: CGAffineTransform(scaleX: scaleX, y: scaleY))
    }

    func convertImageToBuffer(_ image: CIImage) -> CVPixelBuffer? {
        let context = CIContext(options: nil)
        let attributes : [NSObject:AnyObject] = [
            kCVPixelBufferCGImageCompatibilityKey : true as AnyObject,
            kCVPixelBufferCGBitmapContextCompatibilityKey : true as AnyObject
        ]
        var pxbuffer: CVPixelBuffer?
        let width = Int(image.extent.size.width)
        let height = Int(image.extent.size.height)
        CVPixelBufferCreate(kCFAllocatorDefault, width, height, kCVPixelFormatType_32ARGB, attributes as CFDictionary?, &pxbuffer)
        let pxbufferUnwrapped = pxbuffer!
        let renderContext = CIContext(options: nil)
        renderContext.render(image, to: pxbufferUnwrapped)
        return pxbufferUnwrapped
    }

    func convertPixelBufferToMultiArray(pixelBuffer: CVPixelBuffer) throws -> MLMultiArray {
        // 1. Create a new MLMultiArray from the pixel buffer
        let imageSide = 256
        let pixelBufferWidth = CVPixelBufferGetWidth(pixelBuffer)
        let pixelBufferHeight = CVPixelBufferGetHeight(pixelBuffer)
        assert(pixelBufferWidth == imageSide && pixelBufferHeight == imageSide, "Input image needs to be \(imageSide)x\(imageSide).")

        // 2. Create an MLMultiArray with the same shape as the input tensor
        let array = try MLMultiArray(shape: [1, 3, 256, 256], dataType: .float32)

        // 3. Copy pixel data into array
        CVPixelBufferLockBaseAddress(pixelBuffer, CVPixelBufferLockFlags(rawValue: 0))
        let pixelData = CVPixelBufferGetBaseAddress(pixelBuffer)
        let sourceRowBytes = CVPixelBufferGetBytesPerRow(pixelBuffer)
        let destination = UnsafeMutableBufferPointer<Float32>(start: array.dataPointer.assumingMemoryBound(to: Float32.self), count: array.count)

        for row in 0..<pixelBufferHeight {
            let sourceRow = pixelData! + row * sourceRowBytes
            let destinationRow = destination.baseAddress! + row * pixelBufferWidth
            memcpy(destinationRow, sourceRow, sourceRowBytes)
        }
        CVPixelBufferUnlockBaseAddress(pixelBuffer, CVPixelBufferLockFlags(rawValue: 0))
        return array
    }


    @IBAction func cameraTapped(_ sender: UIBarButtonItem) {
        
        present(imagePicker, animated: true, completion: nil)
    }
    
}

