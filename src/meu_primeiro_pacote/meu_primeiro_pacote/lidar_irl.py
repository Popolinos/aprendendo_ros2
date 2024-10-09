import rclpy
from rclpy.node import Node

from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist, Vector3

from rclpy.qos import QoSProfile, QoSReliabilityPolicy
from tf_transformations import euler_from_quaternion

from collections import deque
from meu_primeiro_pacote import lidar_to_grid_map as lg
from sklearn.cluster import DBSCAN

import math
import numpy as np
import matplotlib.pyplot as plt
from math import cos, sin, radians, pi, acos, asin, log, exp
#escala do mapa virtual em relação ao real
Escala = 4
# Distância mínima para considerar um novo ponto como diferente (0.5 metros)
dist_threshold = 0.4
mapped_points = set()  # Usaremos um conjunto para armazenar os pontos mapeados

#probabilidades para a leitura do lidar
prob_clear = 0.6 #probabilidade de apontar livre quando livre
prob_not_clear_but = 0.4 #probabilidade de apontar um obstáculo quando está livre

prob_not_clear = 0.7 #probabilidade de apontar um obstáculo quando realmente tem um obstáculo
prob_clear_but = 0.3 #probabilidade de apontar livre quando tem obstáculo


class R2D2(Node):

    def __init__(self):
        super().__init__('R2D2')
        self.get_logger().debug('Defined node name as "R2D2"')

        qos_profile = QoSProfile(depth=10, reliability=QoSReliabilityPolicy.BEST_EFFORT)

        # Subscriber for LaserScan
        self.laser = None
        self.create_subscription(LaserScan, '/scan', self.listener_callback_laser, qos_profile)

        # Subscriber for Odometry
        self.pose = None
        self.create_subscription(Odometry, '/odom', self.listener_callback_odom, qos_profile)

        # Publisher for robot control
        self.pub_cmd_vel = self.create_publisher(Twist, '/cmd_vel', 10)
    
    def listener_callback_laser(self, msg):
        self.laser = msg.ranges

    def listener_callback_odom(self, msg):
        self.pose = msg.pose.pose
        
    
    def lidar_read(self):   #O zero graus do laser, está localizado na esquerda do robô
        angles = []
        distances = []
        for i in range (180):
            distances.append(self.laser[i])
        #angles = np.linspace(3*pi/2, -pi/2,360 ) + self.pose.orientation
        angles = np.linspace(0, 2*pi,360 )
        distances = np.array(distances)
        #distances = np.pad(distances,(0,180),mode='constant',constant_values=0)
        return angles, distances
    
    def angulo_grau(self):

        lista_orientacao = [self.pose.orientation.x, self.pose.orientation.y, self.pose.orientation.z, self.pose.orientation.w]

        # Converta o quaternion para ângulos de Euler (roll, pitch, yaw)
        _,_, yaw = euler_from_quaternion(lista_orientacao)

        # Converta o yaw de radianos para graus
        angulo_global = math.degrees(yaw) # yaw, angulo em radianos

        # Ajuste para o intervalo de 0 a 360 graus
        #if angulo_global < 0:
        #    angulo_global += 360
        return angulo_global,yaw

    def mapa_global(self,dist,yaw,map1,ax2,fig):
        #Posição do robô no mapa global 
        x = 10 + self.pose.position.x
        y = 10 - self.pose.position.y

        ang = np.linspace(0, pi,180) # recrio os ângulos, só que dessa vez com os valores que o lidar realmente pega, para criar o mapa global sem um rastro vermelho
        ang += yaw # A leitura do lidar fica alinhada com o theta global do robô 
        # Remover todos os elementos que são iguais a 0 do array dist, removendo os 180 valores zerados
        dist_filtered = dist[dist != 0]

        ox = (np.sin(ang) * dist_filtered  + x ) * Escala 
        oy = (np.cos(ang) * dist_filtered  + y ) * Escala 

        #Ajustes para o plot de ocupação:
        rob_pos = [int(x*Escala),int(y*Escala)]
        global prob_clear, prob_not_clear,prob_clear_but, prob_not_clear_but
        for ox_val, oy_val in zip(ox, oy): 
            if(int(ox_val)>=80):
                ox_val = 79
            if(int(oy_val)>=80):
                oy_val = 79
            line = lg.bresenham((rob_pos[1], rob_pos[0]), (int(oy_val),int(ox_val)))
            for l in line:
                map1[l[0]][l[1]] = 1- (prob_clear*(1-map1[l[0]][l[1]]))/(prob_clear*(1-map1[l[0]][l[1]]) + prob_clear_but*map1[l[0]][l[1]])    #O robô só vai detectar o ponto (int(oy_val),int(ox_val)), o resto fica livre
            #self.get_logger().info(f'ox_val: {int(ox_val)} oy_val:{int(oy_val)}')
            map1[int(oy_val)][int(ox_val)] = (prob_not_clear*map1[l[0]][l[1]])/(prob_not_clear*map1[l[0]][l[1]] + prob_not_clear_but*(1-map1[l[0]][l[1]])) # Dupla leitura de obstáculo para compensar lá em cima que disse que não tinham obstáculos no caminho
            map1[int(oy_val)][int(ox_val)] = (prob_not_clear*map1[l[0]][l[1]])/(prob_not_clear*map1[l[0]][l[1]] + prob_not_clear_but*(1-map1[l[0]][l[1]]))
        #------------------------------------------------------------------------------

        return ox, oy, x, y

    def run(self):
            
            #self.get_logger().debug ('Executando uma iteração do loop de processamento de mensagens.')
            rclpy.spin_once(self)

            #Define as mensagens de controle do robô
            self.ir_para_tras = Twist(linear=Vector3(x= -0.4,y=0.0,z=0.0),angular=Vector3(x=0.0,y=0.0,z= 0.0))
            self.ir_para_frente = Twist(linear=Vector3(x= 0.4,y=0.0,z=0.0),angular=Vector3(x=0.0,y=0.0,z= 0.0))
            self.parar          = Twist(linear=Vector3(x= 0.0,y=0.0,z=0.0),angular=Vector3(x=0.0,y=0.0,z= 0.0))
            self.girar_direita = Twist(linear=Vector3(x= 0.0,y=0.0,z=0.0),angular=Vector3(x=0.0,y=0.0,z= -0.2))
            self.girar_esquerda = Twist(linear=Vector3(x= 0.0,y=0.0,z=0.0),angular=Vector3(x=0.0,y=0.0,z= 0.2))
            
            self.pub_cmd_vel.publish(self.parar)
            rclpy.spin_once(self)

            self.get_logger().info ('Entrando no loop princial do nó.')
            
            # Create a figure with two subplots (one for a plot, one for an image)
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

            #Criando o mapa
            map1 = np.ones((20*Escala, 20*Escala)) * 0.5 # cada quadrado equivale a 0.125 metros, no simulador cada quadrado equivalia a 1 metro

            while(rclpy.ok):
                rclpy.spin_once(self)

                self.get_logger().debug ('Atualizando as distancias lidas pelo laser.')

                if(self.laser == None):
                    continue
                self.distancia_direita   = min((self.laser[  0: 80])) # -90 a -10 graus
                self.distancia_frente    = min((self.laser[ 80:100])) # -10 a  10 graus
                self.distancia_esquerda  = min((self.laser[100:180])) #  10 a  90 graus
                
                if self.pose is None:
                    continue  

                #Lê o lidar
                ang,dist = self.lidar_read()

                #Plota o gráfico com a leitura do lidar

                ox = np.sin(ang) * dist
                oy = np.cos(ang) * dist
                ax1.cla() # limpa o gráfico
                ax1.axis("equal")
                ax1.plot([oy, np.zeros(np.size(oy))], [ox, np.zeros(np.size(oy))],'r-') # lines from 0,0 to the
                #bottom, top = plt.ylim()  # return the current ylim
                #plt.ylim((top, bottom)) # rescale y axis, to match the grid orientation
                ax1.grid(True)

                #Pego o ângulo global
                angulo_global,angulo_global_rad = self.angulo_grau()
                
                #Pego as posições dos pontos em que o lidar bateu
                ox,oy,pos_x,pos_y = self.mapa_global(dist,angulo_global_rad,map1,ax2,fig)
                
                #Desenha o movimento do robô no mapa
                ax2.plot((10 + self.pose.position.x)*Escala,(10 - self.pose.position.y)*Escala, marker='o', linestyle='None', markersize=1,color='red')

                ax2.grid(True)
                ax2.imshow(map1)
                #ax2.colorbar()
                plt.pause(0.1)
                
                #map1 = np.ones((20*Escala, 20*Escala)) * 0.5

                #Lei de controle do robô  
                #if (self.laser[0]<1):
                #    self.pub_cmd_vel.publish(self.girar_direita)
                #elif (self.laser[0]>4):
                #    self.pub_cmd_vel.publish(self.girar_esquerda)
                #elif (self.laser[90]<1.3):
                #    self.pub_cmd_vel.publish(self.girar_esquerda)
                #else:
                #    self.pub_cmd_vel.publish(self.ir_para_frente)                   
                #self.pub_cmd_vel.publish(self.girar_direita)

                #Imprime a posição geral do robô
                self.get_logger().info(f'X: {self.pose.position.x} Y:{self.pose.position.y} Theta: {angulo_global}')
                #self.get_logger().info(f'Map1[79][56]: {map1[79][56]}') # Ponto de obstáculo
                #self.get_logger().info(f'Map1[40][50]: {map1}') # Ponto livre

                rclpy.spin_once(self)
        
    # Destrutor do nó
    def __del__(self):
        self.pub_cmd_vel.publish(self.parar)
        self.get_logger().info('Finalizando o nó! Tchau, tchau...')

# Função principal
def main(args=None):
    rclpy.init(args=args)
    node = R2D2()
    try:
        node.run()
        node.destroy_node()
        rclpy.shutdown()
    except KeyboardInterrupt:
        pass
   
if __name__ == '__main__':
    main()   