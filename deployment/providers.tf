terraform {
  required_providers {
    kubernetes = {
      source = "hashicorp/kubernetes"
      version = "2.11.0"
    }
  }
  required_version = ">= 0.1.0"
}
provider "kubernetes" {
  /* host = minikube_cluster.docker-minikube.host
  client_certificate     = minikube_cluster.docker-minikube.client_certificate
  client_key             = minikube_cluster.docker-minikube.client_key
  cluster_ca_certificate = minikube_cluster.docker-minikube.cluster_ca_certificate */
  config_context         = "minikube"
  config_path            = "~/.kube/config"
}

/* provider "minikube" {} */

/* resource "minikube_cluster" "docker-minikube" {
  cluster_name = "terraform-provider-minikube-acc-docker"
  nodes        = 3
} */

/* provider "kubernetes" {
  host = minikube_cluster.docker-minikube.host

  client_certificate     = minikube_cluster.docker-minikube.client_certificate
  client_key             = minikube_cluster.docker-minikube.client_key
  cluster_ca_certificate = minikube_cluster.docker-minikube.cluster_ca_certificate
} */
