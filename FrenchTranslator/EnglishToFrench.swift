//
//  EnglishToFrench.swift
//  FrenchTranslator
//
//  Created by Karl McPhee on 10/2/23.
//

import UIKit

class EnglishToFrench: UIViewController {

    @IBOutlet weak var translationLabel: UILabel!
    @IBOutlet weak var EnglishText: UITextField!
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
    }
    

    @IBAction func EngTranslateClicked(_ sender: Any) {
        var uploadText: String = ""
        if EnglishText.text == "" {
            return
        } else {
            uploadText = EnglishText.text!
        }
        let url = URL(string: apiEndpoint+"/translateFromEng")
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
                        print(jsonResponse)
                        
                        DispatchQueue.main.async {
                            let translatedText = jsonResponse["Translation"] as? String ?? "Failed to translate text"
                            self.translationLabel.text = translatedText
                            print(jsonResponse["Translation"]!)
                            print("hi")
                                
                          //  }
                        }
                    } catch {
                        print("error")
                    }
                }
            }
        }
        task.resume()
    }
    /*
    // MARK: - Navigation

    // In a storyboard-based application, you will often want to do a little preparation before navigation
    override func prepare(for segue: UIStoryboardSegue, sender: Any?) {
        // Get the new view controller using segue.destination.
        // Pass the selected object to the new view controller.
    }
    */

}

struct uploadData: Codable {
    var text: String
}
