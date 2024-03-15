//
//  FrenchToEnglish.swift
//  FrenchTranslator
//
//  Created by Karl McPhee on 3/9/24.
//

import UIKit

class FrenchToEnglish: UIViewController {

    @IBOutlet weak var displayText: UILabel!
    @IBOutlet weak var inputText: UITextField!
    var apiEndpoint = ""
    override func viewDidLoad() {
        super.viewDidLoad()
        
        if let path = Bundle.main.path(forResource: "Info", ofType: "plist") {
            if let dict = NSDictionary(contentsOfFile: path) as? [String: Any] {
                if let value = dict["ParseClientKey"] as? String {
                    apiEndpoint = value
                }
                }
            }

        // Do any additional setup after loading the view.
    }
    

  
     @IBAction func onClick(_ sender: Any) {
         
         var uploadText: String
         if inputText.text == "" {
             return
         } else {
             uploadText = inputText.text!
         }
         
         let url = URL(string: apiEndpoint+"/translateFromFrench")
         let parameters: [String: Any] = [
             "request" : [
                 "phrase": uploadText,
                 "password": "pwd"]]
         
         var request = URLRequest(url: url!)
         request.httpMethod="POST"
         request.setValue("Application/json", forHTTPHeaderField: "Content-Type")
         guard let httpBody = try? JSONSerialization.data(withJSONObject: parameters, options: []) else {
             return
         }
         request.httpBody = httpBody
         request.timeoutInterval = 20
         
         let session = URLSession.shared
         
         let task = session.dataTask(with: request) { (data, response, error ) in
             if error != nil {
                 let alert = UIAlertController(title: "Error", message: error?.localizedDescription, preferredStyle: UIAlertController.Style.alert)
                 let okButton = UIAlertAction(title: "OK", style: UIAlertAction.Style.default, handler: nil)
                 alert.addAction(okButton)
                 self.present(alert, animated: true, completion: nil)
             } else {
                 if data != nil {
                     do {
                         let jsonResponse = try JSONSerialization.jsonObject(with: data!, options: JSONSerialization.ReadingOptions.mutableContainers) as! Dictionary<String, Any>
                         //print(jsonResponse)
                         
                         DispatchQueue.main.async {
                             let translatedText = jsonResponse["Translation"] as? String ?? "Failed to translate text"
                             self.displayText.text = translatedText
                         }
                     } catch {
                         print("error")
                     }
                 }
             }
         }
         task.resume()
     }
    
}
