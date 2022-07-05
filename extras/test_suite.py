import unittest

class TestIndividual(unittest.TestCase):
    
    def setUp(self):
        self.ind_up = Individual(moves=[0]*16, position=[0,0]) #individual that always goes up
        self.ind_dx = Individual(moves=[1]*16, position=[0,0]) #always right
        self.ind_dw = Individual(moves=[2]*16, position=[0,0]) #always down
        self.ind_sx = Individual(moves=[3]*16, position=[0,0]) #always left
        #my_Env = Environment() #empty enviroment
    
    #Test movimento nelle 4 direzioni
    def test_Movement(self):
        for x in range(dim+3):
            self.ind_up.move_eat([])
            self.ind_dx.move_eat([])
            self.ind_dw.move_eat([])
            self.ind_sx.move_eat([])
            self.assertEqual(self.ind_up.position, [0,(x+1)%dim])
            self.assertEqual(self.ind_dx.position, [(x+1)%dim,0])
            self.assertEqual(self.ind_dw.position, [0,(-x-1)%dim])
            self.assertEqual(self.ind_sx.position, [(-x-1)%dim,0])
            
    def test_Energy(self):
        self.assertEqual(self.ind_up.get_fitness(),energy)      #checks starting energy
        self.ind_up.move_eat([])                                #moves without eating
        self.assertEqual(self.ind_up.get_fitness(),energy-1)    #checks energy is lowered
        self.ind_up.move_eat([[0,2]])                           #moves on food
        self.assertEqual(self.ind_up.get_fitness(),energy)      #checks energy is increased
        
        
    def test_Cases(self):
        self.assertEqual(self.ind_up.get_case([]), [0,0,0,0])
        self.assertEqual(self.ind_up.get_case([[0,1]]), [1,0,0,0])
        self.assertEqual(self.ind_up.get_case([[0,1],[0,9]]), [1,0,1,0])
        self.assertEqual
    
    def test_Mate(self):
        a = Individual([0]*10)
        b = Individual([1]*10)
        c = a.mate(b,nodes=[3,7])
        self.assertEqual(c,[0,0,0,1,1,1,1,0,0,0])
        self.assertEqual(c.energy,10)
        
    def test_Score(self):
        self.assertEqual( self.ind_up.score()*14 , 7 )
        self.ind_up[1]=3
        self.assertEqual( self.ind_up.score()*14 , 8)
        self.ind_up[2]=2
        self.assertEqual( self.ind_up.score()*14 , 9)
        self.ind_up[4]=1
        self.assertEqual( self.ind_up.score()*14 , 10)
        self.ind_up[12]=1
        self.assertEqual( self.ind_up.score()*14 , 10)
        

if __name__ == '__main__':
    unittest.main()