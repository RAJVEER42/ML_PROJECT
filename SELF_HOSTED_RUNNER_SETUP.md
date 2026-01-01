# Self-Hosted GitHub Actions Runner Setup for EC2

## Quick Setup Steps

### 1. SSH into EC2
```bash
ssh -i your-key.pem ubuntu@your-ec2-ip
```

### 2. Install Required Tools
```bash
sudo apt-get update
sudo apt-get install -y docker.io awscli curl jq
sudo systemctl start docker
sudo systemctl enable docker
sudo usermod -aG docker ubuntu
```
**Logout and login again** after adding user to docker group

### 3. Get Runner Token from GitHub
Go to: `Your Repo → Settings → Actions → Runners → New self-hosted runner`

Copy the commands shown (they include your unique token)

### 4. Setup Runner on EC2
```bash
# Create folder
mkdir actions-runner && cd actions-runner

# Download runner (check GitHub for latest version)
curl -o actions-runner-linux-x64-2.311.0.tar.gz -L \
  https://github.com/actions/runner/releases/download/v2.311.0/actions-runner-linux-x64-2.311.0.tar.gz

# Extract
tar xzf ./actions-runner-linux-x64-2.311.0.tar.gz

# Configure (paste the token from GitHub)
./config.sh --url https://github.com/YOUR_USERNAME/ML_PROJECT --token YOUR_TOKEN

# Install and start as service
sudo ./svc.sh install
sudo ./svc.sh start

# Check status
sudo ./svc.sh status
```

### 5. Configure AWS CLI on EC2
```bash
aws configure
# Enter AWS Access Key ID
# Enter AWS Secret Access Key
# Enter region: eu-north-1
# Enter output format: json
```

Or attach an IAM role to EC2 with ECR permissions (better practice)

### 6. Verify Runner is Active
Check GitHub: `Settings → Actions → Runners` - should show **green/online**

## Common Mistakes

❌ **Not adding user to docker group** - Runner can't execute docker commands
   - Fix: `sudo usermod -aG docker ubuntu` then logout/login

❌ **AWS credentials not configured** - Can't login to ECR
   - Fix: Run `aws configure` or attach IAM role to EC2

❌ **Security group not allowing runner to reach GitHub** - Runner stays offline
   - Fix: Allow outbound HTTPS (443) in security group

❌ **Forgetting to start runner as service** - Runner stops when SSH session ends
   - Fix: Always use `sudo ./svc.sh install` and `sudo ./svc.sh start`

❌ **Using expired token** - Configuration fails
   - Fix: Tokens expire in 1 hour, generate new one from GitHub

❌ **Docker not running** - All deployment steps fail
   - Fix: `sudo systemctl start docker && sudo systemctl enable docker`

❌ **Wrong region in workflow vs AWS CLI** - ECR login fails
   - Fix: Match region in workflow (eu-north-1) with AWS CLI config

## Quick Checks

```bash
# Is runner service running?
sudo ./svc.sh status

# Is docker working without sudo?
docker ps

# Can access ECR?
aws ecr describe-repositories --region eu-north-1

# Is runner showing in GitHub?
# Check: Settings → Actions → Runners → Should be green
```

## Restart Runner if Needed
```bash
cd ~/actions-runner
sudo ./svc.sh stop
sudo ./svc.sh start
```

## Remove Runner (if needed)
```bash
cd ~/actions-runner
sudo ./svc.sh stop
sudo ./svc.sh uninstall
./config.sh remove --token YOUR_REMOVAL_TOKEN
```
