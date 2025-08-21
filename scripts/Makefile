
ansible-setup:
	pip install ansible
	ansible-galaxy role install geerlingguy.postgresql


#
# Inventory: to run remotely
# ---
# 
# database:
#   hosts:
#     192.168.2.157:
#       ansible_user: setepenre
    
#
# Inventory: to run on localhost
# ---
# 
# database:
#   hosts:
#     localhost:

ansible-deploy:
	ansible-playbook -v playbook.yaml -i inventory.yaml -K
