//
//  ViewController.swift
//  MLtest
//
//  Created by Jade Choghari on 28/06/2023.
//

import UIKit
import CoreML
import Vision
//import PythonKit
import Accelerate
import Foundation
import Surge

class ViewController: UIViewController, UIImagePickerControllerDelegate, UINavigationControllerDelegate {
    
    var imgWidth = 0
    var imgHeight = 0
    var inputImage: UIImage? = nil
    @IBOutlet weak var imageView: UIImageView!
    let imagePicker = UIImagePickerController()
    
    @IBOutlet weak var predictionLabel: UILabel!
    
    override func viewDidLoad() {
        super.viewDidLoad()
        // Do any additional setup after loading the view.
        
        imagePicker.delegate = self
        imagePicker.sourceType = .photoLibrary
        imagePicker.allowsEditing = false
    }
    
    func imagePickerController(_ picker: UIImagePickerController, didFinishPickingMediaWithInfo info: [UIImagePickerController.InfoKey : Any]) {
        
        if let userPickedImage = info[UIImagePickerController.InfoKey.originalImage] as? UIImage {
//            imageView.image = userPickedImage
            imgWidth = Int(userPickedImage.size.width)
            imgHeight = Int(userPickedImage.size.height)
            inputImage = userPickedImage
            print("The widht of the imnage: ", imgWidth)
            print("The height of the image: ", imgHeight)
           processImage(image: userPickedImage)
           let segmentationResult = processImageWithSegmentation(image: userPickedImage)
            print("the seg results", segmentationResult)
            // Draw rectangles on the image
            drawRectanglesOnImage(image: userPickedImage, boxes: segmentationResult)

        }
        imagePicker.dismiss(animated: true, completion: nil)
    }
    func drawRectanglesOnImage(image: UIImage, boxes: [[Decimal]]) {
        // Create a graphics context of the original image
        UIGraphicsBeginImageContextWithOptions(image.size, false, 0.0)
        image.draw(in: CGRect(x: 0, y: 0, width: image.size.width, height: image.size.height))
        
        // Create a graphics context for drawing rectangles
        guard let context = UIGraphicsGetCurrentContext() else {
            return
        }
        
        // Set rectangle properties
        let lineWidth: CGFloat = 2.0
        let strokeColor = UIColor.green.cgColor
        
        // Draw rectangles on the image
        for box in boxes {
            if box.count >= 6 {
                let x1 = CGFloat(truncating: box[0] as NSNumber)
                let y1 = CGFloat(truncating: box[1] as NSNumber)
                let x2 = CGFloat(truncating: box[2] as NSNumber)
                let y2 = CGFloat(truncating: box[3] as NSNumber)
                
                let rect = CGRect(x: x1, y: y1, width: x2 - x1, height: y2 - y1)
                
                context.setStrokeColor(strokeColor)
                context.setLineWidth(lineWidth)
                context.addRect(rect)
                context.strokePath()
                
            }
        }
        print("Drawing")
//         Get the image with drawn rectangles
        guard let drawnImage = UIGraphicsGetImageFromCurrentImageContext() else {
            return
        }
        print("This is the image:", drawnImage)
        // Display the image with drawn rectangles
//        imageView.image = drawnImage
        // End the graphics context
        UIGraphicsEndImageContext()
    }
     
// Uncomment this if u want to get the highest name with the highest prob
//    func processImage(image: UIImage) {
//        // Resize the image to 256x256 pixels
//        guard let resizedImage = image.resize(to: CGSize(width: 256, height: 256)),
//              let buffer = resizedImage.toMLMultiArray() else {
//            print("Resizing or converting to pixel buffer failed")
//            return
//        }
//
//        // Load the ML model
//        do {
//            let config = MLModelConfiguration()
//            let model = try EfficientNet_b3(configuration: config)
//            let input = EfficientNet_b3Input(x_1: buffer)
//
//            // Make the prediction
//            let output = try model.prediction(input: input)
//
//            // Process the model's output here.
//            let outputArray = outputArrayFromMLMultiArray(output: output.var_1837)
//            let maxScoreIndex = outputArray.indices.max(by: { outputArray[$0] < outputArray[$1] })
//            let predictedDiseaseOrActivity = diseaseOrActivityForIndex(maxScoreIndex)
//            print("Predicted Disease or Activity: \(predictedDiseaseOrActivity)")
//
//        } catch {
//            print("Error loading the model: \(error)")
//        }
//    }

    func outputArrayFromMLMultiArray(output: MLMultiArray) -> [Float] {
        let count = output.count
        var array = [Float](repeating: 0, count: count)
        for i in 0..<count {
            array[i] = output[i].floatValue
        }
        return array
    }
// Uncomment this as well if u want to get the highest name with the highest prob
//    func diseaseOrActivityForIndex(_ index: Int?) -> String {
//        // Replace this with your own mapping from indices to diseases or activities
//        // based on how you've trained your model
//        let idxToClass: [Int: String] = [
//                0: "amalgam tattoo",
//                1: "aphthus ulcer",
//                2: "candidiasis1",
//                3: "fibroma",
//                4: "geographic tongue",
//                5: "hemangioma",
//                6: "leukoplakia-hyperkeratosis",
//                7: "lichen planus",
//                8: "melanotic macule",
//                9: "mucocele",
//                10: "papilloma",
//                11: "pyogenic granuloma",
//                12: "squamous cell carcinoma",
//                13: "tongue cheek chewing"
//            ]
//
//        guard let index = index, let diseaseOrActivity = idxToClass[index] else {
//            return "Unknown"
//        }
//
//        return diseaseOrActivity
//    }
    
    //Comment this function if u want to get the highest name with the highest prob
    func processImage(image: UIImage) {
        // Resize the image to 256x256 pixels
        guard let resizedImage = image.resize(to: CGSize(width: 256, height: 256)),
              let buffer = resizedImage.toMLMultiArray() else {
            print("Resizing or converting to pixel buffer failed")
            return
        }

        // Load the ML model
        do {
            let config = MLModelConfiguration()
            let model = try EfficientNet_b3(configuration: config)
            let input = EfficientNet_b3Input(x_1: buffer)

            // Make the prediction
            let output = try model.prediction(input: input)

            // Process the model's output here.
            let outputArray = outputArrayFromMLMultiArray(output: output.var_1837)

            // Compute softmax to get probabilities
            let exps = outputArray.map { exp($0) }
            let sum = exps.reduce(0, +)
            let probabilities = exps.map { $0 / sum }

            // Get top 3 classes
            var classProbabilities = zip((0..<outputArray.count), probabilities).sorted(by: { $0.1 > $1.1 })
            classProbabilities = Array(classProbabilities.prefix(3))

            // Extract the actual classes and probabilities
            let topClasses = classProbabilities.map { diseaseOrActivityForIndex($0.0) }
            let topP = classProbabilities.map { $0.1 }

            print(topClasses, topP)
            DispatchQueue.main.async {
                let alert = UIAlertController(title: "Top Predictions", message: "Here are the top predictions for the image", preferredStyle: .alert)
                
                for i in 0..<3 {
                    //                        let action = UIAlertAction(title: "\(topClasses[i]): \(topP[i])", style: .default)
                    //                        alert.addAction(action)
                    let probabilityPercentage = String(format: "%.2f%%", topP[i] * 100)
                    let actionTitle = "\(topClasses[i].capitalized): \(probabilityPercentage)"
                    let action = UIAlertAction(title: actionTitle, style: .default) { _ in
                        self.predictionLabel.text = actionTitle
                    }
                    alert.addAction(action)
                }
                
                    let cancelAction = UIAlertAction(title: "OK", style: .cancel)
                    alert.addAction(cancelAction)
                    
                    self.present(alert, animated: true)
                }

        } catch {
            print("Error loading the model: \(error)")
        }
    }
//
    func diseaseOrActivityForIndex(_ index: Int) -> String {
        // Replace this with your own mapping from indices to diseases or activities
        // based on how you've trained your model
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

        return idxToClass[index, default: "Unknown"]
    }
    
    // function of the second model
//    func processImageWithSegmentation(image: UIImage) {
//        // Resize the image to 640x640 pixels
//            guard let resizedImage = image.resize(to: CGSize(width: 640, height: 640)) else {
//                print("Resizing image failed")
//                return
//            }
//
//            // Convert resized image to CVPixelBuffer of type kCVPixelFormatType_32BGRA
//            guard let buffer = resizedImage.toPixelBuffer(format: kCVPixelFormatType_32BGRA) else {
//                print("Converting to pixel buffer failed")
//                return
//            }
//
//        // Load the ML model
//        do {
//            let config = MLModelConfiguration()
//            let model = try yolov8l_seg(configuration: config)
//            let input = yolov8l_segInput(image: buffer)
//
//            // Make the prediction
//            let output = try model.prediction(input: input)
//
//            // Check if the output feature value exists and can be converted to MLMultiArray
//            guard let outputMultiArray = output.featureValue(for: "var_1506")?.multiArrayValue else {
//                print("Failed to convert output feature value to MLMultiArray.")
//                return
//            }
//
//            guard let outputP = output.featureValue(for: "p")?.multiArrayValue else {
//                print("Failed to convert output feature value to MLMultiArray.")
//                return
//            }
////             Print the first few values
//            let outputValues = Array(UnsafeBufferPointer(start: outputMultiArray.dataPointer.bindMemory(to: Float.self, capacity: outputMultiArray.count), count: outputMultiArray.count))
//            print(outputValues.prefix(10))
//
////             Also print the shape of the MLMultiArray
//            print("Shape of var_1506: \(outputMultiArray.shape)")
//
//            let outputValuesP = Array(UnsafeBufferPointer(start: outputP.dataPointer.bindMemory(to: Float.self, capacity: outputP.count), count: outputP.count))
//            print(outputValuesP.prefix(10))
//
//            print("Shape of p: \(outputP.shape)")
//            // Convert the MLMultiArray to CGImage
////            guard let cgImage = try? MLMultiArrayToCGImage(output: outputMultiArray) else {
////                print("Failed to convert MLMultiArray to CGImage.")
////                return
////            }
////
////            // Create a UIImage from the CGImage
////            let outputImage = UIImage(cgImage: cgImage)
////
////            // Display the output image
////            DispatchQueue.main.async {
////                self.imageView.image = outputImage
////            }
//
//        } catch {
//            print("Error loading the model: \(error)")
//        }
//    }
    func processImageWithSegmentation(image: UIImage) -> [[Decimal]]{
        // Resize the image to 640x640 pixels
        guard let resizedImage = image.resize(to: CGSize(width: 640, height: 640)) else {
            print("Resizing image failed")
            return []
        }
        
        // Convert resized image to CVPixelBuffer of type kCVPixelFormatType_32BGRA
        guard let buffer = resizedImage.toPixelBuffer(format: kCVPixelFormatType_32ARGB) else {
            print("Converting to pixel buffer failed")
            return []
        }
        
        
        // Load the ML model
        do {
            let config = MLModelConfiguration()
            let model = try best_2(configuration: config)
            let input = best_2Input(image: buffer)

            // Make the prediction
            let output = try model.prediction(input: input)
            
            // shape of var [1, 37, 8400]
            let output0 = output.var_1504
            //shape of p [1, 32, 160, 160]
            let output1 = output.p
            
            print("the shape of output0", output0.shape)
//             Convert MLMultiArray to a normal Swift array
            let array = convertMultiArrayToArray(output0)
            print("the shape after conversion: ", array[0][0].count)
            

//            // Perform dimension reduction
            let reducedArray = array[0]
            print("reduced !", reducedArray[0].count)
            
            let transposedArray = transpose(reducedArray)
//            print("Transposed ", transposedArray[0])
            
            print("Output0", transposedArray.count, transposedArray[0].count)
            let fourArray = convertMultiArrayToArray4d(output1) // 3 x 4 x 2
            
            print("The shape of ouput1", fourArray.count, fourArray[0].count, fourArray[0][0].count, fourArray[0][0][0].count)
            // Perform dimension reduction
            let reducedArray1 = fourArray[0]
            print("Output1", reducedArray1.count, reducedArray1[0].count, reducedArray1[0][0].count)
//            let test12 = parseRow(row: transposedArray[0])
//            print("result of the first parse row[0]: ", test12)
//            let boxes = transposedArray
//                .map { parseRow(row: $0) }
//                .filter { $0[5] > 1.2 }
            
//            let result = nonMaxSuppression(boxes: boxes, iouThreshold: 0.9999999)
////            print("processed!:", result)
            ///
//            let boxes = transposedArray
//                .map { parseRow(row: $0) }
//                .filter { $0[5] > 1.3 }
            
//            let boxes = transposedArray.compactMap { parseRow(row: $0) }
//                .filter { $0[5] > 1.3 }
            let boxes = transposedArray
                .map { parseRow(row: $0) }
                .filter { $0[5] > 2.25 }
            
           
//            print("Count of boxes", boxes)
            
            // If something goes wrong delete here
            
            //Start step 1
            func convertMultiArrayToArray4d(_ multiArray: MLMultiArray) -> [[[[Decimal]]]] {
                let shape = multiArray.shape.map { $0.intValue }
                var currentIndex = [NSNumber](repeating: 0, count: shape.count)
                var array = [[[[Decimal]]]](repeating: [[[Decimal]]](repeating: [[Decimal]](repeating: [Decimal](repeating: 0, count: shape[3]), count: shape[2]), count: shape[1]), count: shape[0])

                for i in 0..<multiArray.count {
                    let value = Decimal(multiArray[currentIndex].doubleValue) // Convert the value to Decimal
                    let indices = currentIndex.compactMap { $0.intValue }

                    array[indices[0]][indices[1]][indices[2]][indices[3]] = value

                    // Update the current index to iterate through all elements
                    for i in (0..<currentIndex.count).reversed() {
                        let currentIndexValue = currentIndex[i].intValue

                        if currentIndexValue < shape[i] - 1 {
                            currentIndex[i] = NSNumber(value: currentIndexValue + 1)
                            break
                        } else {
                            currentIndex[i] = 0
                        }
                    }
                }

                return array
            }
            
            
            
            //end of step 1
            // Assuming output0 is a 2D array in Swift
            // Create a sample 2D array (output0) // 3 x 10

            let boxes1 = transposedArray.map { Array($0[0..<5]) }
            let masks1 = transposedArray.map { Array($0[5...]) }
            
            print("boxes1", boxes1.count, boxes1[0].count)
            print("masks1", masks1.count, masks1[0].count)
            // End of step 2
            // Step 3
            //Step 3 reshape output1 to (32, 160x160 )
            // 3 x 4 x2
            
            print("Reduced array;", reducedArray1[0].count)
            let reshapedArray = reducedArray1.map { subArray in
                return subArray.flatMap { $0 }
            }

            print("Reshape count: 160x160 ", reshapedArray.count, reshapedArray[0].count) // here mustsee 160x160

            //End of step3
            //Step 4
            // Define matrix dimensions
            
//            print("Answer number: ", rowsAT)
//            // Create result matrix
//            var resultMatrixT = Array(repeating: Array(repeating: Decimal(0.0), count: columnsBT), count: rowsAT)
//
//            print("Multiplication Started!")
//            let concurrentQueue = DispatchQueue(label: "com.example.concurrentMatrixMultiplication", attributes: .concurrent)
//
//            DispatchQueue.concurrentPerform(iterations: rowsAT) { i in
//                for j in 0..<columnsBT {
//                    for k in 0..<columnsAT {
//                        resultMatrixT[i][j] += masks1[i][k] * reshapedArray[k][j]
//                        print("OK !", i, j, k)
//
//                    }
//                }
//            }
            func multiplyMatrices(_ matrixA: [[Decimal]], _ matrixB: [[Decimal]]) -> [[Decimal]]? {
                let rowsA = matrixA.count
                let columnsA = matrixA[0].count
                let rowsB = matrixB.count
                let columnsB = matrixB[0].count

                // Check if matrices can be multiplied
                guard columnsA == rowsB else {
                    return nil
                }

                // Convert matrices to flat arrays of Double
                var flatMatrixA = matrixA.flatMap { $0.map { Double($0 as NSNumber) } }
                var flatMatrixB = matrixB.flatMap { $0.map { Double($0 as NSNumber) } }

                // Create result matrix
                var result = [Double](repeating: 0, count: rowsA * columnsB)

                // Perform matrix multiplication using Accelerate framework
                vDSP_mmulD(flatMatrixA, 1, flatMatrixB, 1, &result, 1, vDSP_Length(rowsA), vDSP_Length(columnsB), vDSP_Length(columnsA))

                // Convert the result back to a 2D array of Decimal
                var resultMatrix = [[Decimal]]()
                
                for i in 0..<rowsA {
                    let startIndex = i * columnsB
                    let endIndex = startIndex + columnsB
                    let row = Array(result[startIndex..<endIndex]).map { Decimal($0) }
                    resultMatrix.append(row)
                    let percentage = Double(i + 1) / Double(rowsA) * 100
                    let remainingPercentage = 100 - percentage
//                    print("Remaining time: \(remainingPercentage)%")
                }

                return resultMatrix
            }
            
            let result = multiplyMatrices(masks1, reshapedArray)
            print("after multip", result!.count, result![0].count)
            
            //
            //// Create result matrix
            //var resultMatrix = [Decimal](repeating: 0.0, count: rowsA * columnsB)
            //
            //// Perform matrix multiplication using Accelerate framework
            //vDSP_mmulD(arrayA, 1, arrayB, 1, &resultMatrix, 1, vDSP_Length(rowsA), vDSP_Length(columnsB), vDSP_Length(columnsA))
            //
            //// Convert result back to 2D array
            //var resultArray = [[Decimal]]()
            //for i in 0..<rowsA {
            //    let startIndex = i * columnsB
            //    let endIndex = startIndex + columnsB
            //    let row = Array(resultMatrix[startIndex..<endIndex])
            //    resultArray.append(row)
            //}
            //
            //print("Here is a test", resultArray)
//            // Perform matrix multiplication
//            for i in 0..<rowsAT {
//                for j in 0..<columnsBT {
//                    for k in 0..<columnsAT {
//                        resultMatrixT[i][j] += masks1[i][k] * reshapedArray[k][j]
////                        print("OK !", i, j, k)
//                    }
//                }
////            }
//
//            print("Here is the result matrix: 1 ", resultMatrixT.count, "results 2", resultMatrixT[0].count)// expected 8400 x 25605
            
            // End of step 4
            //Step 5
            //perform additions
            var combinedMask: [[Decimal]] = []
            
            // for (box, mask) in zip(reshapedArray, masks1)
            //or (box, mask) in zip(boxes1, result!)
            for (box, mask) in zip(boxes1, result!) {
                let combinedRow = box + mask
                combinedMask.append(combinedRow)
            }
            
            print("Combined worked! ", combinedMask.count, combinedMask[0].count)
            //stop
            //print("Combination tEST:", combined[0].count)
            //End of step5
            //Step 6 mask iterations

            // Masks iterations
//            let boxesTC: [[Double]] = [
//                [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0],
//                [12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0]
//            ]
//
//            var masksTC: [[Decimal]] = []
            let img_width: Double = 1170
            let img_height: Double = 1516
//            let resultTest = [129.041128125, 477.5784921875, 1080.81894375, 1404.9874653125, 20, 2.412757]
            print("Starting with masks!")
            
//            let boxesMask: [ParsedRow] = combinedMask.map { parseRowMask(row: $0) }
            let boxesMask: [ParsedRow] = combinedMask.enumerated().map { parseRowMask(index: $0.offset, row: $0.element) }

            
            print("test box maks ", boxesMask[8249].mask)
            let maskImage = convertToUIImage(mask: boxesMask[8249].mask)
            let newSize = CGSize(width: Int(round(NSDecimalNumber(decimal: boxesMask[8249].x2 - boxesMask[8249].x1).doubleValue)),
                                 height: Int(round(NSDecimalNumber(decimal: boxesMask[8249].y2 - boxesMask[8249].y1).doubleValue)))
            
            let resizedImage = resizeMask(image: maskImage!, targetSize: newSize)
            let color = UIColor.green
            let finalImage = overlayMask(baseImage: inputImage!, mask: maskImage!, color: color)
            //ANA
            imageView.image = finalImage
            
            print("Started step 1")
            let boxesMaskFiltered = boxesMask.filter { $0.maxProbability > 11 }
            print("started step 2")
            let boxesMaskDecimals: [[Decimal]] = boxesMaskFiltered.map { [$0.x1, $0.y1, $0.x2, $0.y2, $0.classId, $0.maxProbability, Decimal($0.index)]}
            print("boxes mask decimal", boxesMaskDecimals.count, boxesMaskDecimals[0].count)
//
            let results1 = nonMaxSuppression(boxes: boxesMaskDecimals, iouThreshold: 0.9)
//            let resultNormal = nonMaxSuppression(boxes: boxes, iouThreshold: 0.7)
//            print("boxes", boxes.count, boxes[0].count)
//            print("result normal", resultNormal.count, resultNormal[0].count)
            print("after non max", results1.count, results1[0].count)
            print("lets see non max", results1)
            
//
//            let TestParsedMask = combinedMask[14684]
//            let parseTC = parseRowMask(row: TestParsedMask)
//            print("Parsed mask", parseTC.mask)
            
            //boxes[234]
//            for row in combinedMask {
//                print("Started!")
//                let mask1 = Array(row[5...25604])
////                masksTC.append(mask1)
//                let reshapedMatrix = reshapeToMatrix(array: mask1, rows: 160, cols: 160)
//                let sigmoidMatrix = sigmoidMatrix(reshapedMatrix)
//
//                let mask_x1 = round(resultTest[0]/img_width*160)
//                let mask_y1 = round(resultTest[1]/img_height*160)
//                let mask_x2 = round(resultTest[2]/img_width*160)
//                let mask_y2 = round(resultTest[3]/img_height*160)
//                let binaryMask = createMask(from: sigmoidMatrix)
//                // Use indexing operation to select the subarray
////                let selectedMask = binaryMask[Int(mask_y1)..<Int(mask_y2)].map { $0[Int(mask_x1)..<Int(mask_x2)] }
//                let selectedMask = binaryMask[Int(mask_y1)..<Int(mask_y2)].map { Array($0[Int(mask_x1)..<Int(mask_x2)]) }
//
//                print("selected mask", selectedMask)
//                let maskImage = convertToUIImage(mask: selectedMask)
//
//                let x1 = Double(mask_x1), x2 = Double(mask_x2)
//                let y1 = Double(mask_y1), y2 = Double(mask_y2)
//
//                let newSize = CGSize(width: round(x2-x1), height: round(y2-y1))
//                let resizedImage = resizeMask(image: maskImage!, targetSize: newSize)
//                let maskArray = imageToArray(image: resizedImage)
//
//                let finalMask = convertToUIImage(mask: maskArray!)
//                print("This is the final image", selectedMask)
//                imageView.image = finalMask!
//                return []
//
//            }
            // HERE COMMENTED THE MASK THINGS BECAUSE WE ARE DOING IT AS GETMASK
//            for row in combinedMask {
//                if row[5...].max()! > 2.4 {
//                    let mask1 = Array(row[5...25604])
//    //                masksTC.append(mask1)
//                    print("mask1", mask1.count)
//                    let reshapedMatrix = reshapeToMatrix(array: mask1, rows: 160, cols: 160)
//                    print("reshaped mas to 160, 160 ->", reshapedMatrix.count, reshapedMatrix[0].count)
//                    let sigmoidMatrix = sigmoidMatrix(reshapedMatrix)
//                    print("data", row[0], row[1], row[2], row[3])
//                    let row0Double = NSDecimalNumber(decimal: row[0]).doubleValue
//                    let row1Double = NSDecimalNumber(decimal: row[1]).doubleValue
//                    let row2Double = NSDecimalNumber(decimal: row[2]).doubleValue
//                    let row3Double = NSDecimalNumber(decimal: row[3]).doubleValue
//                    let mask_x1 = round(row0Double/img_width*160)
//                    let mask_y1 = round(row1Double/img_height*160)
//                    let mask_x2 = round(row2Double/img_width*160)
//                    let mask_y2 = round(row3Double/img_height*160)
//                    let binaryMask = createMask(from: sigmoidMatrix)
//                    // Use indexing operation to select the subarray
//    //              let selectedMask = binaryMask[Int(mask_y1)..<Int(mask_y2)].map { $0[Int(mask_x1)..<Int(mask_x2)] }
//                    let selectedMask = binaryMask[Int(mask_y1)..<Int(mask_y2)].map { Array($0[Int(mask_x1)..<Int(mask_x2)]) }
////                    print("selected mask", selectedMask)
//                    let maskImage = convertToUIImage(mask: selectedMask)
//
//                    let testImage = convertToUIImage(mask: binaryMask)
//                    let x1 = Double(mask_x1), x2 = Double(mask_x2)
//                    let y1 = Double(mask_y1), y2 = Double(mask_y2)
//
//                    let newSize = CGSize(width: round(x2-x1), height: round(y2-y1))
//                    let resizedImage = resizeMask(image: maskImage!, targetSize: newSize)
//                    let maskArray = imageToArray(image: resizedImage)
//
//                    let finalMask = convertToUIImage(mask: maskArray!)
////                    print("This is the final image", selectedMask)
////                    imageView.image = finalMask!
//                    imageView.image = finalMask
//                    return []
//                }
//
//            }
            //TILL HERE
            //all these functons are related to masks
            func imageToArray(image: UIImage) -> [[UInt8]]? {
                guard let cgImage = image.cgImage else { return nil }

                let width = cgImage.width
                let height = cgImage.height

                let bitsPerComponent = 8
                let bytesPerRow = width
                let totalBytes = height * bytesPerRow

                var pixelValues = [UInt8](repeating: 0, count: totalBytes)
                let colorSpace = CGColorSpaceCreateDeviceGray()

                guard let context = CGContext(data: &pixelValues,
                                              width: width,
                                              height: height,
                                              bitsPerComponent: bitsPerComponent,
                                              bytesPerRow: bytesPerRow,
                                              space: colorSpace,
                                              bitmapInfo: CGImageAlphaInfo.none.rawValue) else { return nil }
                
                context.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))

                var matrix = [[UInt8]]()
                for x in 0..<height {
                    var row = [UInt8]()
                    for y in 0..<width {
                        let val = pixelValues[(x * width) + y]
                        row.append(val)
                    }
                    matrix.append(row)
                }

                return matrix
            }

            func convertToUIImage(mask: [[UInt8]]) -> UIImage? {
//                print("This is the mask", mask)
//                print("Mask", mask.count, mask[0].count)
                let width = mask[0].count
                let height = mask.count

                let rawData = mask.flatMap { $0 } // Flatten your 2D array
                let cfbuffer = CFDataCreate(nil, rawData, rawData.count)!
                let dataProvider = CGDataProvider(data: cfbuffer)!
                let colorSpace = CGColorSpaceCreateDeviceGray()
                let bitmapInfo = CGBitmapInfo(rawValue: CGImageAlphaInfo.none.rawValue)

                if let cgImage = CGImage(width: width, height: height, bitsPerComponent: 8, bitsPerPixel: 8, bytesPerRow: width,
                                         space: colorSpace, bitmapInfo: bitmapInfo, provider: dataProvider,
                                         decode: nil, shouldInterpolate: false, intent: .defaultIntent) {
                    return UIImage(cgImage: cgImage)
                } else {
                    return nil
                }
            }
            func resizeMask(image: UIImage, targetSize: CGSize) -> UIImage {
                let size = image.size

                let widthRatio  = targetSize.width  / size.width
                let heightRatio = targetSize.height / size.height

                // Figure out what our orientation is, and use that to form the rectangle
                var newSize: CGSize
                if(widthRatio > heightRatio) {
                    newSize = CGSize(width: size.width * heightRatio, height: size.height * heightRatio)
                } else {
                    newSize = CGSize(width: size.width * widthRatio, height: size.height * widthRatio)
                }

                // This is the rect that we've calculated out and this is what is actually used below
                let rect = CGRect(x: 0, y: 0, width: newSize.width, height: newSize.height)

                // Actually do the resizing to the rect using the ImageContext stuff
                UIGraphicsBeginImageContextWithOptions(newSize, false, 1.0)
                image.draw(in: rect)
                let newImage = UIGraphicsGetImageFromCurrentImageContext()
                UIGraphicsEndImageContext()

                return newImage!
            }
            //stop functions related to mask

            //print("Testing mask iteration: ", masksTC)
//            func applyThreshold(toMatrix matrix: [[Decimal]], threshold: Decimal) -> [[UInt8]] {
//                let binaryMatrix = matrix.map { row in
//                    return row.map { value in
//                        return value > threshold ? 255 : 0
//                    }
//                }
//                return binaryMatrix
//            }
//
//            func cropMask(_ mask: [[UInt8]], x1: Decimal, y1: Decimal, x2: Decimal, y2: Decimal) -> [[UInt8]] {
//                let croppedMask = Array(mask[y1..<y2]).map { row in
//                    return Array(row[x1..<x2])
//                }
//                return croppedMask
//            }

            
            //End of step6
            //Step 7
            func reshapeToMatrix(array: [Decimal], rows: Int, cols: Int) -> [[Decimal]] {
                var matrix: [[Decimal]] = []
                
                for i in 0..<rows {
                    let startIndex = i * cols
                    let endIndex = startIndex + cols
                    let row = Array(array[startIndex..<endIndex])
                    matrix.append(row)
                }
                
                return matrix
            }
            
            // to 160 x 160
//            let reshapedMasks = masksTC.map { row in
//                return reshapeToMatrix(array: row, rows: 160, cols: 160)
//            }
//
//            // Example usage:
//            print(reshapedMasks)
            //end of step 7
            //Step 8 Applying sigmoid

//            let matrix: [[Decimal]] = [
//                [Decimal(1.8), Decimal(2.0), Decimal(3.0)],
//                [Decimal(4.9), Decimal(5.0), Decimal(6.0)],
//                [Decimal(7.0), Decimal(8.0), Decimal(9.0)]
//            ]

            func sigmoid(_ x: Decimal) -> Decimal {
                let doubleX = NSDecimalNumber(decimal:x).doubleValue
                let result = 1.0 / (1.0 + exp(-doubleX))
                return Decimal(result)
            }

            func sigmoidMatrix(_ matrix: [[Decimal]]) -> [[Decimal]] {
                return matrix.map { row in
                    return row.map { element in
                        sigmoid(element)
                    }
                }
            }
            
            func createMask(from matrix: [[Decimal]]) -> [[UInt8]] {
                let threshold: Decimal = 0.5 //!!!!!!
                let white: UInt8 = 255
                let black: UInt8 = 0

                return matrix.map { row in
                    row.map { element in
                        return element > threshold ? white : black
                    }
                }
            }
            // End of step 8
            //stop
        // Select the index of the choosing row after filtering > 2.25
//            let selectedIndices = transposedArray
//                .enumerated()
//                .filter { parseRow(row: $0.element)[5] > 2.25 }
//                .map { $0.offset }
//
//            print("Hope they are correct: ", selectedIndices)
//            let boxesTest = transposedArray
//                .map { parseRow(row: $0) }
//                .filter { $0[5] > 2.25 }
            
//            //superimp
//            let results = nonMaxSuppression(boxes: boxes, iouThreshold: 0.7)
//
//            print("The shape of the non max after: ", results[0])
//            return results //sspi
            func convertMultiArrayToArray(_ multiArray: MLMultiArray) -> [[[Decimal]]] {
                let shape = multiArray.shape.map { $0.intValue }
                var currentIndex = [NSNumber](repeating: 0, count: shape.count)
                var array = [[[Decimal]]](repeating: [[Decimal]](repeating: [Decimal](repeating: 0.0, count: shape[2]), count: shape[1]), count: shape[0])

                for i in 0..<multiArray.count {
                    let value = multiArray[currentIndex].decimalValue
                    let indices = currentIndex.compactMap { $0.intValue }

                    array[indices[0]][indices[1]][indices[2]] = value

                    // Update the current index to iterate through all elements
                    for i in (0..<currentIndex.count).reversed() {
                        let currentIndexValue = currentIndex[i].intValue

                        if currentIndexValue < shape[i] - 1 {
                            currentIndex[i] = NSNumber(value: currentIndexValue + 1)
                            break
                        } else {
                            currentIndex[i] = 0
                        }
                    }
                }

                return array
            }
            
            func transpose<T>(_ array: [[T]]) -> [[T]] {
                guard let rowCount = array.first?.count else {
                    return []
                }
                
                var transposedArray: [[T]] = Array(repeating: Array(repeating: array[0][0], count: array.count), count: rowCount)
                
                for (i, row) in array.enumerated() {
                    for (j, element) in row.enumerated() {
                        transposedArray[j][i] = element
                    }
                }
                
                return transposedArray
            }
//
            
            
            
            func parseRow(row: [Decimal]) -> [Decimal] {
                // Extracting the values using array slicing
                let extractedValues = Array(row[..<4])

                // Assigning variables of type Decimal
                var xc: Decimal = 0, yc: Decimal = 0, w: Decimal = 0, h: Decimal = 0

                if extractedValues.count >= 4 {
                    xc = extractedValues[0]
                    yc = extractedValues[1]
                    w = extractedValues[2]
                    h = extractedValues[3]
                }

                let img_width: Decimal = 1170
                let img_height: Decimal = 1516

                // Example values for xc, yc, w, h (using Decimal type)
                let x1 = (xc - w/2) / 640 * img_width
                let y1 = (yc - h/2) / 640 * img_height
                let x2 = (xc + w/2) / 640 * img_width
                let y2 = (yc + h/2) / 640 * img_height
                
                
                // Finding the maximum probability value and its index
                var maxProbability: Decimal = 0
                var classId: Int = 0

                for (index, value) in row.enumerated() where index >= 4 {
                    if value > maxProbability {
                        maxProbability = value
                        classId = index
                    }
                }
                
                // Creating an array with the desired values
                let result: [Decimal] = [x1, y1, x2, y2, Decimal(classId), maxProbability]
                
                return result
            }
            
            func get_mask(row: [Decimal], box: (Decimal, Decimal, Decimal, Decimal)) -> [[UInt8]] {
                let mask1 = Array(row[5...25604])
                let reshapedMatrix = reshapeToMatrix(array: mask1, rows: 160, cols: 160)
                let sigmoidMatrix = sigmoidMatrix(reshapedMatrix)
                
                let x1 = box.0
                let y1 = box.1
                let x2 = box.2
                let y2 = box.3
                
                let mask_x1 = max(0, min(159, Int(round((NSDecimalNumber(decimal: x1).doubleValue / NSDecimalNumber(decimal: Decimal(img_width)).doubleValue) * 160))))
                let mask_y1 = max(0, min(159, Int(round((NSDecimalNumber(decimal: y1).doubleValue / NSDecimalNumber(decimal: Decimal(img_height)).doubleValue) * 160))))
                let mask_x2 = max(0, min(159, Int(round((NSDecimalNumber(decimal: x2).doubleValue / NSDecimalNumber(decimal: Decimal(img_width)).doubleValue) * 160))))
                let mask_y2 = max(0, min(159, Int(round((NSDecimalNumber(decimal: y2).doubleValue / NSDecimalNumber(decimal: Decimal(img_height)).doubleValue) * 160))))
           
                // Perform the mask selection
              
                let binaryMask = createMask(from: sigmoidMatrix)
                print("this is the mask")
                var selectedMask = [[UInt8]]()
                for i in mask_y1..<mask_y2 - 1 {
                    let row = Array(binaryMask[i][mask_x1..<mask_x2])
                    selectedMask.append(row)
                }
                
                
//                let selectedMask = binaryMask[Int(mask_y1)..<Int(mask_y2)].map { Array($0[Int(mask_x1)..<Int(mask_x2)]) }
                
//                let maskImage = convertToUIImage(mask: selectedMask)
                
//                let newSize = CGSize(width: Int(round(NSDecimalNumber(decimal: x2 - x1).doubleValue)),
//                                     height: Int(round(NSDecimalNumber(decimal: y2 - y1).doubleValue)))
//                let resizedImage = resizeMask(image: maskImage!, targetSize: newSize)
//
//                imageView.image = resizedImage
//                let maskArray = imageToArray(image: resizedImage)
//                return maskArray!
                return selectedMask
            }
            // lets see this method because it needs to be of type decimal not any
            struct ParsedRow {
                let x1: Decimal
                let y1: Decimal
                let x2: Decimal
                let y2: Decimal
                let classId: Decimal
                let maxProbability: Decimal
                let mask: [[UInt8]]
                let index: Int
            }
            func parseRowMask(index: Int, row: [Decimal]) -> ParsedRow {
                // Extracting the values using array slicing
                let extractedValues = Array(row[..<4])

                // Assigning variables of type Decimal
                var xc: Decimal = 0, yc: Decimal = 0, w: Decimal = 0, h: Decimal = 0

                if extractedValues.count >= 4 {
                    xc = extractedValues[0]
                    yc = extractedValues[1]
                    w = extractedValues[2]
                    h = extractedValues[3]
                }

                let img_width: Decimal = 1170
                let img_height: Decimal = 1516

                // Example values for xc, yc, w, h (using Decimal type)
                let x1 = (xc - w/2) / 640 * img_width
                let y1 = (yc - h/2) / 640 * img_height
                let x2 = (xc + w/2) / 640 * img_width
                let y2 = (yc + h/2) / 640 * img_height

                // Finding the maximum probability value and its index
                var maxProbability: Decimal = 0
                var classId: Int = 0

                for (index, value) in row.enumerated() where index >= 4 {
                    if value > maxProbability {
                        maxProbability = value
                        classId = index
                    }
                }
                let box: (Decimal, Decimal, Decimal, Decimal) = (x1, y1, x2, y2)
                let mask = get_mask(row: row, box: box)
                // Creating an array with the desired values
                
                return ParsedRow(x1: x1, y1: y1, x2: x2, y2: y2, classId: Decimal(classId), maxProbability: maxProbability, mask: mask, index: index)
            }
            
            
            
//            print("boxes: ", boxes)
            
            
            func overlayMask(baseImage: UIImage, mask: UIImage, color: UIColor) -> UIImage? {
                // Create a new image context
                UIGraphicsBeginImageContextWithOptions(baseImage.size, false, baseImage.scale)
                
                // Draw the base image
                baseImage.draw(in: CGRect(origin: .zero, size: baseImage.size))

                // Set the blend mode and color for the mask
                color.set()
                
                // Calculate the position for the mask
                let maskRect = CGRect(x: (baseImage.size.width - mask.size.width) / 2,
                                      y: (baseImage.size.height - mask.size.height) / 2,
                                      width: mask.size.width,
                                      height: mask.size.height)
                
                // Draw the mask
                mask.draw(in: maskRect, blendMode: .normal, alpha: 0.5)
                
                // Retrieve the resulting image
                let result = UIGraphicsGetImageFromCurrentImageContext()
                
                // Clean up the image context
                UIGraphicsEndImageContext()
                
                return result
            }

            func intersection(box1: [Decimal], box2: [Decimal]) -> Decimal {
                let box1_x1 = box1[0], box1_y1 = box1[1], box1_x2 = box1[2], box1_y2 = box1[3]
                let box2_x1 = box2[0], box2_y1 = box2[1], box2_x2 = box2[2], box2_y2 = box2[3]
                
                let x1 = max(box1_x1, box2_x1)
                let y1 = max(box1_y1, box2_y1)
                let x2 = min(box1_x2, box2_x2)
                let y2 = min(box1_y2, box2_y2)
                
                let intersectionArea = (x2 - x1) * (y2 - y1)
                return intersectionArea
            }
            func union(box1: [Decimal], box2: [Decimal]) -> Decimal {
                let box1_x1 = box1[0], box1_y1 = box1[1], box1_x2 = box1[2], box1_y2 = box1[3]
                let box2_x1 = box2[0], box2_y1 = box2[1], box2_x2 = box2[2], box2_y2 = box2[3]
                
                let box1_area = (box1_x2 - box1_x1) * (box1_y2 - box1_y1)
                let box2_area = (box2_x2 - box2_x1) * (box2_y2 - box2_y1)
                
                let intersectionArea = intersection(box1: box1, box2: box2)
                
                let unionArea = box1_area + box2_area - intersectionArea
                return unionArea
            }
            func iou(box1: [Decimal], box2: [Decimal]) -> Decimal {
                let intersectionArea = intersection(box1: box1, box2: box2)
                let unionArea = union(box1: box1, box2: box2)
                
                let iouValue = intersectionArea / unionArea
                return iouValue
            }
            func nonMaxSuppression(boxes: [[Decimal]], iouThreshold: Decimal) -> [[Decimal]] {
                var sortedBoxes = boxes.sorted { $0[5] > $1[5] }
                
                var result: [[Decimal]] = []

                while sortedBoxes.count > 0 {
                    let currentBox = sortedBoxes[0]
                    result.append(currentBox)
                    
                    sortedBoxes = sortedBoxes.filter {
                        iou(box1: currentBox, box2: $0) < iouThreshold
                    }
                }
                
                return result
            }
            func nonMaxSuppressionMask(boxes: [ParsedRow], iouThreshold: Decimal) -> [ParsedRow] {
                var sortedBoxes = boxes.sorted { $0.maxProbability > $1.maxProbability }

                var result: [ParsedRow] = []

                while sortedBoxes.count > 0 {
                    let currentBox = sortedBoxes[0]
                    result.append(currentBox)

                    sortedBoxes = sortedBoxes.filter {
                        iou(box1: [currentBox.x1, currentBox.y1, currentBox.x2, currentBox.y2],
                            box2: [$0.x1, $0.y1, $0.x2, $0.y2]) < iouThreshold
                    }
                }

                return result
            }


        } catch {
            print("Error loading the model: \(error)")
        }
        print("Error in predictions")
        return []
    }

    

    func MLMultiArrayToCGImage(output: MLMultiArray) throws -> CGImage? {
        let height = output.shape[0].intValue
        let width = output.shape[1].intValue
        var bufferPointer = output.dataPointer

        let byteCount = width * height * 4
        let bitmapInfo = CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue)
            .union(.byteOrder32Big)

        guard let context = CGContext(
            data: nil,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: width * 4,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: bitmapInfo.rawValue)
            else {
            return nil
        }

        guard let buffer = context.data else { return nil }

        let pixelBuffer = buffer.bindMemory(to: UInt8.self, capacity: byteCount)

        for y in 0..<height {
            for x in 0..<width {
                let pixel = bufferPointer.assumingMemoryBound(to: Float.self)
                let offset = y * width * 4 + x * 4
                pixelBuffer[offset] = UInt8(pixel[0] * 255)     // red error
                pixelBuffer[offset+1] = UInt8(pixel[1] * 255)   // green
                pixelBuffer[offset+2] = UInt8(pixel[2] * 255)   // blue
                pixelBuffer[offset+3] = 0xFF                    // alpha
                bufferPointer = bufferPointer.advanced(by: 1)
            }
        }

        guard let cgImage = context.makeImage() else {
            return nil
        }
        return cgImage
    }


    @IBAction func cameraTapped(_ sender: UIBarButtonItem) {
        
        present(imagePicker, animated: true, completion: nil)
    }
    
}
//extension UIImage {
//    func resize(to size: CGSize) -> UIImage? {
//        UIGraphicsBeginImageContextWithOptions(size, false, self.scale)
//        self.draw(in: CGRect(origin: .zero, size: size))
//
//        let resizedImage = UIGraphicsGetImageFromCurrentImageContext()
//        UIGraphicsEndImageContext()
//
//        return resizedImage
//    }
//
//    func toMLMultiArray() -> MLMultiArray? {
//        guard let cgImage = self.cgImage else {
//            return nil
//        }
//
//        let width = cgImage.width
//        let height = cgImage.height
//        let pixelData = cgImage.dataProvider!.data
//        let bytesPerRow = cgImage.bytesPerRow
//        let numBytes = bytesPerRow * height
//
//        var imagePixels: [Float] = Array(repeating: 0, count: width * height * 3)
//        let pixelValues = CFDataGetBytePtr(pixelData)
//
//        for i in stride(from: 0, to: Int(numBytes), by: bytesPerRow) {
//            for j in stride(from: 0, to: bytesPerRow, by: 4) {
//                let index = i / 4 + j / 4
//                let r = Float(pixelValues![i + j + 2]) / 255.0
//                let g = Float(pixelValues![i + j + 1]) / 255.0
//                let b = Float(pixelValues![i + j]) / 255.0
//                imagePixels[index] = r
//                imagePixels[index + width * height] = g
//                imagePixels[index + 2 * width * height] = b
//            }
//        }
//
//        let shape = [NSNumber(value: 1), NSNumber(value: 3), NSNumber(value: height), NSNumber(value: width)]
//        guard let imageMultiArray = try? MLMultiArray(shape: shape as [NSNumber], dataType: .float32) else {
//            return nil
//        }
//
//        for (index, pixelValue) in imagePixels.enumerated() {
//            imageMultiArray[index] = NSNumber(value: pixelValue)
//        }
//
//        return imageMultiArray
//    }
//
//}
extension UIImage {
    func resize(to size: CGSize) -> UIImage? {
        UIGraphicsBeginImageContextWithOptions(size, false, self.scale)
        self.draw(in: CGRect(origin: .zero, size: size))

        let resizedImage = UIGraphicsGetImageFromCurrentImageContext()
        UIGraphicsEndImageContext()

        return resizedImage
    }

    func toMLMultiArray() -> MLMultiArray? {
        guard let cgImage = self.cgImage else {
            return nil
        }

        let width = cgImage.width
        let height = cgImage.height
        let pixelData = cgImage.dataProvider!.data
        let bytesPerRow = cgImage.bytesPerRow
        let numBytes = bytesPerRow * height

        let means: [Float] = [0.485, 0.456, 0.406]
        let stds: [Float] = [0.229, 0.224, 0.225]

        var imagePixels: [Float] = Array(repeating: 0, count: width * height * 3)
        let pixelValues = CFDataGetBytePtr(pixelData)

        for i in stride(from: 0, to: Int(numBytes), by: bytesPerRow) {
            for j in stride(from: 0, to: bytesPerRow, by: 4) {
                let index = i / 4 + j / 4
                let r = Float(pixelValues![i + j + 2]) / 255.0
                let g = Float(pixelValues![i + j + 1]) / 255.0
                let b = Float(pixelValues![i + j]) / 255.0
                imagePixels[index] = (r - means[0]) / stds[0]
                imagePixels[index + width * height] = (g - means[1]) / stds[1]
                imagePixels[index + 2 * width * height] = (b - means[2]) / stds[2]
            }
        }

        let shape = [NSNumber(value: 1), NSNumber(value: 3), NSNumber(value: height), NSNumber(value: width)]
        guard let imageMultiArray = try? MLMultiArray(shape: shape as [NSNumber], dataType: .float32) else {
            return nil
        }

        for (index, pixelValue) in imagePixels.enumerated() {
            imageMultiArray[index] = NSNumber(value: pixelValue)
        }

        return imageMultiArray
    }
    func toPixelBuffer(format: OSType) -> CVPixelBuffer? {
        let attrs = [kCVPixelBufferCGImageCompatibilityKey: kCFBooleanTrue,
             kCVPixelBufferCGBitmapContextCompatibilityKey: kCFBooleanTrue] as CFDictionary
        var pixelBuffer: CVPixelBuffer?
        let status = CVPixelBufferCreate(kCFAllocatorDefault,
                                         Int(self.size.width),
                                         Int(self.size.height),
                                         format,
                                         attrs,
                                         &pixelBuffer)
        guard status == kCVReturnSuccess else {
            return nil
        }
        
        CVPixelBufferLockBaseAddress(pixelBuffer!, CVPixelBufferLockFlags(rawValue: 0))
        let pixelData = CVPixelBufferGetBaseAddress(pixelBuffer!)
        
        let rgbColorSpace = CGColorSpaceCreateDeviceRGB()
        guard let context = CGContext(data: pixelData,
                                      width: Int(self.size.width),
                                      height: Int(self.size.height),
                                      bitsPerComponent: 8,
                                      bytesPerRow: CVPixelBufferGetBytesPerRow(pixelBuffer!),
                                      space: rgbColorSpace,
                                      bitmapInfo: CGImageAlphaInfo.noneSkipFirst.rawValue)
        else {
            return nil
        }
        
        context.translateBy(x: 0, y: self.size.height)
        context.scaleBy(x: 1.0, y: -1.0)
        
        UIGraphicsPushContext(context)
        self.draw(in: CGRect(x: 0, y: 0, width: self.size.width, height: self.size.height))
        UIGraphicsPopContext()
        
        CVPixelBufferUnlockBaseAddress(pixelBuffer!, CVPixelBufferLockFlags(rawValue: 0))
        
        return pixelBuffer
    }

}
