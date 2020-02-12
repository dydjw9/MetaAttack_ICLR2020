#Utntargeted 
#python test_all.py --attacked_model 3 --total_number 1000 --untargeted True --istransfer True --max_fintune_iter 25 --learning_rate 0.6e-2
#python test_all.py --attacked_model 4 --total_number 1000 --untargeted True --istransfer True --max_fintune_iter 25 --learning_rate 0.6e-2
#python test_all.py --attacked_model 3 --total_number 1000 --untargeted True --istransfer False --max_fintune_iter 25 --learning_rate 0.6e-2
#python test_all.py --attacked_model 4 --total_number 1000 --untargeted True --istransfer False --max_fintune_iter 25 --learning_rate 0.6e-2

#Targeted
python test_all.py --attacked_model 3 --total_number 1000 --untargeted False --istransfer True
python test_all.py --attacked_model 4 --total_number 1000 --untargeted False --istransfer True
python test_all.py --attacked_model 3 --total_number 1000 --untargeted False --istransfer False
python test_all.py --attacked_model 4 --total_number 1000 --untargeted False --istransfer False

