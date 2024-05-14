# TMRL - Remote Training Setup

To train more efficiently, we can make use of the LIACS GPU servers. There is a few steps we need to take to get access
to these servers more easily and set up an SSH tunnel to them. This document will guide you through the process of 
setting up everything SSH related. 

The servers we're trying to connect to are:

* **Vibranium** - 2 NVIDIA RTX 3090 (24GB ea)
* **Duranium** - 6 NVIDIA GTX 980Ti (6GB ea), 2 NVIDIA Titan X (12GB ea)

Setup can be done for both, so multiple people can train models on different machines. 

---
## Table of Contents
1. [SSH Setup](#ssh-setup)
    1. [SSH Config](#ssh-config)
    2. [Setup](#setup)
        1. [First install](#first-install)
        2. [Generating keypair](#generating-keypair)
        3. [Copying to remote server(s)](#copying-to-remote-servers)
2. [Remote Setup](#remote-setup)
    1. [Setting up video cards](#setting-up-video-cards)


----
## SSH Setup

### SSH Config

The SSH config file can be used to connect to specified servers with preconfigured commands. This saves a lot of time 
when logging into a remote machine and executing commands on a remote device. 

Copy the text in the first code block below into your `~\.ssh\config` file, which is usually located in `C:\Users\<username>\.ssh\config` on 
Windows. Replace `<LU Username>` with your own Leiden University username, `<ssh keyname>` with the name of the SSH key you 
have created, and `<Win Username>` with your Windows username. The `IdentityFile` should point to the private key, while the public key should be copied to the 
`~/.ssh/authorized_keys` file on the remote server.

```commandline
# Global settings if not explicitly specified otherwise.
Host *
    user <LU Username>
    PreferredAuthentications publickey,password
    IdentityFile C:\Users\<Win Username>\.ssh\<Private Key>
    AddKeysToAgent yes

# SSH Gateway to get access to the Leiden University network.
Host sshgw.leidenuniv.nl
    HostName sshgw.leidenuniv.nl
    Port 22

# Hop server used to get access to the LIACS servers.

# The hop server with all its aliases
Host ssh.liacs.nl silver.liacs.nl silver
    Hostname ssh.liacs.nl
    # MACs were specified, because connection was not possible otherwise. I used `ssh -Q mac` and then did 
    # `ssh -m hmac-sha2-512 <LU username>@ssh.liacs.nl` to find the correct MACs used by the remote server.
    # Reference: https://stackoverflow.com/questions/60108032/corrupted-mac-on-input-ssh-dispatch-run-fatalmessage-authentication-code-incor
    MACs hmac-sha2-256-etm@openssh.com,hmac-sha2-512-etm@openssh.com,umac-128-etm@openssh.com
    Port 22


### GPU Servers ###
# Using `ssh vibranium` or `ssh duranium` will directly connect to the GPU servers, as long as ssh.liacs.nl is in the config file. 

# GPU server LIACS - 6 NVIDIA GTX 980Ti (6GB ea), 2 NVIDIA Titan X (12GB ea)
Host duranium
    Hostname duranium.liacs.nl
    ProxyJump ssh.liacs.nl
    Port 22

# GPU server LIACS - 2 NVIDIA RTX 3090 (24GB ea)
Host vibranium
    Hostname vibranium.liacs.nl
    ProxyJump ssh.liacs.nl
    Port 22


# Adding a tunnel to the GPU servers (for MGAIA).

# Explanation:
# - ProxyCommand ssh -W %h:%p ssh.liacs.nl sets up a tunnel to the remote server <servername>.liacs.nl using ssh.liacs.nl 
# as an intermediary server
# - LocalForward 1234 <servername>.liacs.nl:22 sets up forwarding of the local port 1234 to the remote server <servername>.liacs.nl on port 22.
Host silver-tunnel
    HostName ssh.liacs.nl
    # ProxyJump ssh.liacs.nl
    LocalForward 1234 duranium:22
    LocalForward 1235 vibranium:22
    MACs hmac-sha2-256-etm@openssh.com,hmac-sha2-512-etm@openssh.com,umac-128-etm@openssh.com
    # This configuration directive tells SSH to terminate the connection attempt if it cannot establish all requested port forwards.
    ExitOnForwardFailure yes
    # This setting configures the interval, in seconds, that the SSH client will wait before sending a null packet to the server to keep the connection alive. 
    # This packet is sent if no other data has been transmitted during this interval.
    ServerAliveInterval 60
    # This directive sets the number of server alive messages (see ServerAliveInterval) which may be sent without SSH receiving any messages back from the server. 
    # If this limit is reached without receiving any response, SSH will terminate the connection.
    ServerAliveCountMax 3

# Run the command `silver-tunnel` to set up a tunnel in one Powershell window. Now, you can run the following commands:
# - `ssh -p 1234 <LU username>@localhost` to connect to duranium
# - `ssh -p 1235 <LU username>@localhost` to connect to vibranium
# or run one `ssh localduranium` and `ssh localvibranium` to connect to the respective servers.

Host localduranium
    HostName localhost
    Port 1234

Host localvibranium
    HostName localhost
    Port 1235


```



----
### Setup

#### First install
For this specific setup, we're making use of Trackmania on Windows. First, we need to make sure that we have 
OpenSSH installed on our Windows machine. Installation instructions can be found [here][openssh-windows].

Once SSH has been installed, make sure it is started automatically when starting a new `PowerShell` Window. This can be 
done, either via Admin Powershell:
```powershell
Set-Service ssh-agent -StartupType 'Automatic'
Start-Service ssh-agent
```

Or via the Services App (see [here][auto-start-ssh-agent] for more information).

#### Generating keypair
Next, we need to generate a keypair. This can be done by running the following command in a `PowerShell` window:
```powershell
ssh-keygen -t ed25519 -C "your_email@example.com" -f C:\Users\YourUsername\.ssh\<key name>
```
Make sure to secure this key with a long passphrase that you save in a password manager (e.g. Bitwarden) for later use.
Also give the key a recognizable name, so that you can easily identify it later on.

Furthermore, make sure to add the key with `ssh-add` on your own machine, so that you don't have to enter the passphrase every time.

#### Copying to remote server(s)

Next, we need to copy the public key to the server. This can be done by running the following command:
```powershell
cat C:\Users\<YourUsername>\.ssh\<key name>.pub | ssh <user>@sshgw.leidenuniv.nl "cat >> ~/.ssh/authorized_keys"
```

Alternatively, you can use the `scp` command to copy the public key to the server and append it to authorized keys:
```powershell
scp C:\Users\<YourUsername>\.ssh\<recognizable key name>.pub <username>@<username>:/home/username/.ssh/
cat ~/.ssh/ed25519_key.pub >> ~/.ssh/authorized_keys
```

Afterwards, ensure that the SSH directory and files have the correct permissions:
```sh
chmod 700 ~/.ssh
chmod 600 ~/.ssh/authorized_keys
```

`chmod 700` ensures that only the owner can read, write and execute the directory, while `chmod 600` ensures that 
only the owner can read and write the file.

Assuming you're still logged into the first remote server (the gateway), you can now use copy the public key 
to the second server (e.g. `vibranium`). This can be done by either repeating the steps above, or by using the 
`ssh-copy-id` command (if available):
```sh
ssh-copy-id <LU username>@vibranium.liacs.nl
ssh-copy-id <LU username>@duranium.liacs.nl
```

----

## Remote Setup 
Once the SSH setup has been completed, you can now connect to the remote servers by running the following command:
```powershell
ssh silver-tunnel 
```
to set up a tunnel in one Powershell window. Then, you can run the following commands to test your connection to each
respective server:
```powershell
ssh -p 1234 <LU username>@localhost
ssh -p 1235 <LU username>@localhost
```

or run one of the following commands to connect to the respective servers:
```powershell
ssh localduranium
ssh localvibranium
```
if this has been set up in your SSH config file. Examples of this are also shown in the 
[Remote Access to REL][remote-access-liacs-rel] document.

Of course, you can change the ports you wish to LocalForward to in your SSH config file, as long as they are not 
already in use. To check which ports are in use, you can run the following command:
```powershell
netstat -a -b
```

### Information on GPUs

You can run the following command locally on the server to see which GPUs are in use:
```sh
nvidia-smi 
```

or remotely by running one of the following commands (substitute `<port>` with the port you've set up in the SSH config file for the respective server):
```sh
ssh -p <port> <LU username>@localhost "nvidia-smi"
ssh localduranium "nvidia-smi"
ssh localvibranium "nvidia-smi"
```


This command outputs the following information:
```
Mon May 13 15:05:50 2024
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 545.23.08              Driver Version: 545.23.08    CUDA Version: 12.3     |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
|                                         |                      |               MIG M. |
|=========================================+======================+======================|
|   0  NVIDIA GeForce RTX 3090        Off | 00000000:3B:00.0 Off |                  N/A |
|100%   74C    P2             336W / 350W |  22938MiB / 24576MiB |     96%      Default |
|                                         |                      |                  N/A |
+-----------------------------------------+----------------------+----------------------+
|   1  NVIDIA GeForce RTX 3090        Off | 00000000:AF:00.0 Off |                  N/A |
| 30%   34C    P8              20W / 350W |    430MiB / 24576MiB |      0%      Default |
|                                         |                      |                  N/A |
+-----------------------------------------+----------------------+----------------------+

+---------------------------------------------------------------------------------------+
| Processes:                                                                            |
|  GPU   GI   CI        PID   Type   Process name                            GPU Memory |
|        ID   ID                                                             Usage      |
|=======================================================================================|
|    0   N/A  N/A     27575      C   python                                    22930MiB |
|    1   N/A  N/A     27575      C   python                                      422MiB |
+---------------------------------------------------------------------------------------+
```

This shows that the first GPU is being used by a Python process with PID 27575. This can be useful to check if the
video cards are in use and being used by your process for example. To find out the PID of the process you're running 
(e.g. Python), you can run the following command:
```sh
ps aux | grep python
```

If you want to use only a single GPU to not hog resources, you can specify the GPU you want to use by setting the `CUDA_VISIBLE_DEVICES` 
through the command:
```commandline
export CUDA_VISIBLE_DEVICES=<GPU ID>
```

This command persists for the lifetime of a terminal session, can be set before a command to run a specific process 
on a specific GPU, or, when using it in a script, persists for the run duration of the script.

### Setting up virtual environment
Since we only need to run the `Trainer` in a remote environment, we do not need to install 
OpenPlanet. Instead, we can just create a virtual environment and install the necessary packages. You can either setup an
SSH key for git, or use `scp` to copy the `requirements.txt` to the server. 

[//]: # (```commandline)

[//]: # (scp ~/path/to/requirements.txt <server name>:/home/<LU username>/path/to/requirements.txt)

[//]: # (```)

Then, you can install the necessary packages by running the following commands:
```sh
# Make a new virtual environment named mgaia.
conda create --name mgaia python=3.11 --verbose

# Activate the virtual environment
conda activate mgaia

# Install the tmrl package
pip install tmrl
# Validate the installation
python -m tmrl --install
```





## References
1. [OpenSSH Windows Installation][openssh-windows]
2. [Auto-start SSH Agent][auto-start-ssh-agent]
3. [Remote Access to LIACS Servers][remote-access-liacs-rel]

[openssh-windows]: https://learn.microsoft.com/en-us/windows-server/administration/openssh/openssh_install_firstuse?tabs=gui
[auto-start-ssh-agent]: https://stackoverflow.com/questions/44203409/how-to-start-ssh-agent-on-windows-automatically
[remote-access-liacs-rel]: https://liacs.leidenuniv.nl/~takesfw/SNACS/liacs-rel-ssh-access.pdf