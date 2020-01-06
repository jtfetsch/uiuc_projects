#include <iostream>
#include <cstdio>
#include <cstring>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <errno.h>
#include <signal.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netdb.h>
#include <unistd.h>
#include "tcpSocket.h"
#include "ftpServer.h"

#define	LINELEN	4096
#define	ARGLEN	1024
#define	HOSTNAMELEN 64

/*
*   SIGCHLD handler
*/
void child_handler(int signum)
{
    //cleans up all terminated child processes
    while(waitpid(-1,NULL,WNOHANG) != -1);
}

void handle_request(int fd)
{
	char args[ARGLEN];
	char request[LINELEN];
	FILE *fpin  = fdopen(fd, "r");
	int received = recv(fd,request,LINELEN,0);
	char cmdPtr[5];
	int index = 0;
	for(;index < 4;++index)
	{
		cmdPtr[index] = request[index];
	}
	cmdPtr[index] = '\0';
	++index;
	int count = index;
	while(request[index] != '\n')
	{
		args[index - count] = request[index];
		++index;
	}
	args[index - count] = '\0';
	++index;
	//fgets(request,LINELEN,fpin);//read request
	/*int index = 0;
	const char delim[]=" \t\r\n";
	char *buff = strtok(request,delim);
	while(buff!=NULL)//parse input request
	{
		args[index] = buff;
		++index;
 		buff = strtok (NULL, delim);
	}*/	
	int pid = fork();
    	if ( pid == -1 ){
		perror("fork");
		return;
	}
	if(pid == 0){//run grep in child process
	    std::string command(cmdPtr);
	    if(command == "getx")
	    {
		ftpServer::receive_get(fd,args);
	    }
	    else if(command == "putx")
	    {
		ftpServer::receive_put(fd,args,request + index,received - index);
	    }
	    else if(command == "delx")
	    {
		ftpServer::receive_delete(fd,args);
	    }
	    else if(command == "list")
	    {
		ftpServer::receive_list(fd,args);
	    }
	   exit(0);		/* child is done	*/
	}
	else{//parent process
	    close(fd); //Close the socket descriptor in parent
	}
}

int main()
{
	signal(SIGCHLD,child_handler);//collect child exit status to avoid zombies
	struct sockaddr_in cli_addr;
	
	int sock_id = Socket::make_server_socket(FTPPORTNUM);
	if ( sock_id == -1 ) {
		fprintf(stderr, "error in making socket");
		exit(1);
	}
	int clilen = sizeof(cli_addr);
	while(1)
	{
		int fd    = accept( sock_id, (struct sockaddr *)&cli_addr 
		                         , (socklen_t *)&clilen );	/* take a call	*/
		if ( fd == -1 )
		{
		    //continue when accept returns due to interruption
		    if(errno == EINTR)
		        continue;
		    else //other error
			    perror("accept");
	 	}
		else
		{
			handle_request(fd);
		}	
	}
	return 0;
}
