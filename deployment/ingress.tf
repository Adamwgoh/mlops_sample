resource "kubernetes_ingress" "mlflow-ingress" {
  metadata {
    name      = "mlflow-ingress"
    namespace = "mlflow"
  }
  spec {
    rule {
      host = "127.0.0.1"
      http {
        path {
          backend {
            service_name = "mlflow-server"
            service_port = 80
          }
          path = "/"
        }
      }
    }
  }
}