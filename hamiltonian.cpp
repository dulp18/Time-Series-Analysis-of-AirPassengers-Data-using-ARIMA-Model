/* Hello, this is my first github commit*/
#include<bits/stdc++.h>
using namespace std;
bool issafe(bool g[5][5],int p,int i,int path[])
{
	if(g[path[p-1]][i]==0)
	return false;
	for(int j=0;j<p;j++)
	{
		if(path[j]==i)
		return false;
	}
	return true;
}
bool hamil(bool g[5][5],int p,int path[])
{
	if(p==5)
	{
		if(g[path[4]][path[0]]==1)
		return true;
		else
		return false;
	}
	for(int i=1;i<5;i++)
	{
		if(issafe(g,p,i,path))
		{
				path[p]=i;
				if(hamil(g,p+1,path)==true)
				return true;
				path[p]=-1;
		}
	//	cout<<path[i];
	}
	return false;
}
int main()
{
	bool g[5][5]={{0, 1, 0, 1, 0},
                      {1, 0, 1, 1, 1},
                      {0, 1, 0, 0, 1},
                      {1, 1, 0, 0, 1},
                      {0, 1, 1, 1, 0}
                     };
                 int path[5];
                 for(int i=0;i<5;i++)
                 path[i]=-1;
                 path[0]=0;                  
                 if(hamil(g,1,path)==false)
                 cout<<"Not Possible"<<endl;
                 for(int i=0;i<5;i++)
				 {
                 	cout<<path[i]<<"   ";
                 }                

}
