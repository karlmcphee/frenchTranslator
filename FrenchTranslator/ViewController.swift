//
//  ViewController.swift
//  FrenchTranslator
//
//  Created by Karl McPhee on 10/1/23.
//

import UIKit

class ViewController: UIViewController {

    override func viewDidLoad() {
        super.viewDidLoad()
        // Do any additional setup after loading the view.
    }

    @IBAction func EngTranslateClicked(_ sender: Any) {
        performSegue(withIdentifier: "EnglishSegue", sender: nil)
    }
    
    @IBAction func FrenchTranslateClicked(_ sender: Any) {
        performSegue(withIdentifier: "FrenchSegue", sender: nil)
    }
}

