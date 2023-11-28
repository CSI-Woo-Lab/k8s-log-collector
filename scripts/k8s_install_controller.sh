sudo apt update -y
# Create the .conf file to load the modules(overlay, br_netfilter) at bootup
cat <<EOF | sudo tee /etc/modules-load.d/crio.conf
overlay
br_netfilter
EOF

sudo modprobe overlay

# br_netfilter 커널 모듈을 즉시 로드. linux 브릿지 네트워크와 관련된 필터링에 사용.
sudo modprobe br_netfilter

# Set up required sysctl params, these persist across reboots.
# iptables가 브릿지 트래픽을 처리할 수 있도록 설정
cat <<EOF | sudo tee /etc/sysctl.d/99-kubernetes-cri.conf
net.bridge.bridge-nf-call-iptables  = 1
net.ipv4.ip_forward                 = 1
net.bridge.bridge-nf-call-ip6tables = 1
EOF

# 설정했던 값 적용
sudo sysctl --system

# Add Docker's official GPG key
sudo apt-get update -y
sudo apt-get install -y ca-certificates curl gnupg
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg

# Add the repository to Apt sources
echo \
  "deb [arch="$(dpkg --print-architecture)" signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  "$(. /etc/os-release && echo "$VERSION_CODENAME")" stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update -y

# install the docker, containerd latest version
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

# Configure containerd and start service
sudo mkdir -p /etc/containerd
sudo containerd config default | sudo tee /etc/containerd/config.toml
sudo systemctl restart containerd

# Add kuberenetes repository
sudo apt-get update -y
sudo apt-get install -y apt-transport-https ca-certificates curl gpg
curl -fsSL https://pkgs.k8s.io/core:/stable:/v1.27/deb/Release.key | sudo gpg --dearmor -o /etc/apt/keyrings/kubernetes-apt-keyring.gpg
echo 'deb [signed-by=/etc/apt/keyrings/kubernetes-apt-keyring.gpg] https://pkgs.k8s.io/core:/stable:/v1.27/deb/ /' | sudo tee /etc/apt/sources.list.d/kubernetes.list

# repository 갱신
sudo apt-get update -y

# kubelet, kubeadm, kubectl 설치
sudo apt-get install -y kubelet kubeadm kubectl 

# 버전 고정
sudo apt-mark hold kubelet kubeadm kubectl

# systemd cgroup 드라이버를 runc에서 사용
sudo vi /etc/containerd/config.toml
# runc.options-SystemdCgroup을 false -> true로 변경

# containerd와 kubelet 재시작
sudo systemctl restart containerd
sudo systemctl restart kubelet

# disable swap
sudo vi /etc/fstab
# /swap.img -> # /swap.img (주석처리)

# disable swap(swap is not supported in k8s v1.27)
sudo swapoff -a

# 포트 허용
sudo apt install -y firewalld
sudo firewall-cmd --permanent --zone=public --add-port=8080/tcp
sudo firewall-cmd --permanent --zone=public --add-port=6443/tcp
sudo firewall-cmd --permanent --zone=public --add-port=9000/tcp
sudo firewall-cmd --permanent --zone=public --add-port=10248/tcp
sudo firewall-cmd --permanent --zone=public --add-port=10250-10255/tcp
sudo firewall-cmd --permanent --zone=public --add-port=2379-2380/tcp
sudo firewall-cmd --reload

sudo kubeadm init --pod-network-cidr 192.168.0.0/16

# kubectl 설정
mkdir -p $HOME/.kube
yes | sudo cp -i /etc/kubernetes/admin.conf $HOME/.kube/config
sudo chown $(id -u):$(id -g) $HOME/.kube/config

# operator cluster에 등록
kubectl apply -f https://raw.githubusercontent.com/projectcalico/calico/v3.26.4/manifests/calico.yaml

# 정상적으로 등록되었는지 확인
# STATUS Ready인지 확인
kubectl get nodes

# worker 등록에 사용될 token 생성 이후 복사
sudo kubeadm token create --print-join-command