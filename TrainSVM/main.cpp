#include "SVMfilter.h"

int main(int argc, char** argv)
{
	ofstream out_PreRec("result.txt");
    for(int i=0;i<5;i++){
		SVMmodel mysvm(i,0,43);
		mysvm.CrossValidAndTest(out_PreRec,0,i);
	}
	out_PreRec.close();
	return 0;
}