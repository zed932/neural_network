// NetworkManager.swift
import Foundation

class NetworkManager {
    static let shared = NetworkManager()
    private var session: URLSession!
    private let serverURL = URL(string: "http://192.168.1.100:5000/upload")! // Замените на ваш IP
    
    private init() {
        let config = URLSessionConfiguration.default
        config.timeoutIntervalForRequest = 10
        config.timeoutIntervalForResource = 30
        session = URLSession(configuration: config)
    }
    
    func sendImage(_ imageData: Data, completion: @escaping (Bool, Error?) -> Void) {
        var request = URLRequest(url: serverURL)
        request.httpMethod = "POST"
        
        // Создаем multipart/form-data запрос
        let boundary = "Boundary-\(UUID().uuidString)"
        request.setValue("multipart/form-data; boundary=\(boundary)", forHTTPHeaderField: "Content-Type")
        
        var body = Data()
        body.append("--\(boundary)\r\n".data(using: .utf8)!)
        body.append("Content-Disposition: form-data; name=\"file\"; filename=\"photo.jpg\"\r\n".data(using: .utf8)!)
        body.append("Content-Type: image/jpeg\r\n\r\n".data(using: .utf8)!)
        body.append(imageData)
        body.append("\r\n--\(boundary)--\r\n".data(using: .utf8)!)
        
        // Отправляем асинхронно
        let task = session.uploadTask(with: request, from: body) { _, response, error in
            if let error = error {
                completion(false, error)
                return
            }
            
            if let httpResponse = response as? HTTPURLResponse {
                completion(httpResponse.statusCode == 200, nil)
            } else {
                completion(false, NSError(domain: "NetworkError", code: 0, userInfo: [NSLocalizedDescriptionKey: "Invalid response"]))
            }
        }
        task.resume()
    }
}
