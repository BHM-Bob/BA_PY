
#include <iostream>
#include <string>
#include <string.h>

using namespace std;

char* GetTime(void)
{
	time_t tim = time(NULL);
	char* pstr = ctime(&tim);
    pstr[24] = ' ';
    return pstr;
}

float GetClock(void)
{
    return (float)(clock()/(float)CLOCKS_PER_SEC);
}

int main(int args, char** argvs)
{
    char* time = GetTime();
    int cmdLen = 0;

    cout << "\nmbapy (CLI Tool, v1.0.0, built at" <<  __DATE__ << " " << __TIME__ << " "  << time << endl;
    cout << GetClock() << ": get " << args << "args" << endl;
    for(int i = 0; i<args; i++)
    {
        cout << GetClock() << " : " << ": arg " << i << " : " << argvs[i] << endl;
        cmdLen += (strlen(argvs[i]) + 1);
    }

    // short cut
    if(args == 1)
        return 0;

    // parse command
    string cmd = "python -m mbapy.scripts";
    if(argvs[1][0] != '-')
        cmd += ".";
    else
        cmd += " ";
    for(int i = 1; i<args; i++)
    {
        string argvs_i = argvs[i];
        for(size_t pos = argvs_i.find("("); pos != string::npos; pos = argvs_i.find("(", pos+2))
            argvs_i.replace(pos, 1, "\\(");
        for(size_t pos = argvs_i.find(")"); pos != string::npos; pos = argvs_i.find(")", pos+2))
            argvs_i.replace(pos, 1, "\\)");
        cmd = cmd + argvs_i + " ";
    }

    // run command
    cout << "\n\n" << GetClock() << " : running command: " << cmd << endl;
    int ret = system(cmd.c_str());
    cout << "\n\n" << GetClock() << " : end command: " << cmd << endl;

    return ret;
}