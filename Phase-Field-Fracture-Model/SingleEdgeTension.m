% Script for a two dimensional plate under tension

close all
clear

global iter
global adder
global cur_angle


idx = 1;
for i =6:6
    iter=i;
    adder = 0.05+0.05*i;
    angles_stored = [45]

    for jj=1:length(angles_stored)
        c = clock
        cur_angle = angles_stored(1,jj);    
        
        addpath('./utils')
        addpath('./example_data')
        addpath('../nurbs/inst')
        
        
        str_save_prev_1 = append(num2str(i), '_Case_');
        str_save_prev_2 = append(str_save_prev_1, num2str(cur_angle));
        str_save = append(str_save_prev_2, '_Angle/');
        folder_save = append('TrainingSet/', str_save);
        mkdir(['TrainingSet/',str_save])


        Input = @() Input_tensilePlate;
        PF.Geometry = @(geometry)initMeshTensile(geometry);
        PF.Boundary = @(PHTelem,geometry,controlPts)initialBC_tensile(PHTelem,geometry);
        PF.History = @(gaussCord,Fract,numberElements,geometry)history_tensile(gaussCord,...
                        Fract,numberElements,geometry);
        PF.Trac = @(stiffUU,tdisp,Integ,dirichlet,file_name)compTreacTension(stiffUU,tdisp,...
                        Integ,dirichlet,file_name);

        first_F = 'TrainingSet/';
        second_F = append(first_F,num2str(i));
        third_F = append(second_F, '_Case/');

        file_name = append(folder_save,'FD-tensile.txt');
        output = fopen(file_name,'w');
        
        disp(output)
        fprintf(output,'%14.6e %14.6e\n',0,0);
        fclose(output);

        order = '1'; 
        degrad = '1';

        if order == '1'
            if degrad == '1'

                PF.StiffUU = @(PHTelem,sizeBasis,numberElements,dgdx,shape,Fract,Mater,...
                    volume,tdisp,geometry)gStiffnessUU(PHTelem,sizeBasis,...
                    numberElements,dgdx,shape,Mater,volume,tdisp,geometry);
                PF.StiffPhiPhi = @(PHTelem,sizeBasis,numberElements,dgdx,shape,Fract,...
                    Mater,volume,geometry,fenerg,tdisp)gStiffnessPhiPhi(PHTelem,...
                    sizeBasis,numberElements,dgdx,shape,Fract,Mater,volume,geometry,fenerg);

                solver_2nd_Custom(Input,PF,file_name);
                %solver_2nd(Input,PF,file_name);
            else

                PF.StiffUU = @(PHTelem,sizeBasis,numberElements,dgdx,shape,Fract,Mater,...
                    volume,tdisp,geometry)gStiffnessUUcubic(PHTelem,sizeBasis,...
                    numberElements,dgdx,shape,Fract,Mater,volume,tdisp,geometry);
                PF.StiffPhiPhi = @(PHTelem,sizeBasis,numberElements,dgdx,shape,Fract,...
                    Mater,volume,geometry,fenerg,tdisp)gStiffnessPhiPhicubic(PHTelem,...
                    sizeBasis,numberElements,dgdx,shape,Fract,Mater,volume,geometry,fenerg,tdisp);

                solver_2nd(Input,PF,file_name);
            end
        else
            if degrad == '1'

                PF.StiffUU = @(PHTelem,sizeBasis,numberElements,dgdx,shape,Fract,Mater,...
                    volume,tdisp,geometry)gStiffnessUU(PHTelem,sizeBasis,...
                    numberElements,dgdx,shape,Mater,volume,tdisp,geometry);
                PF.StiffPhiPhi = @(PHTelem,sizeBasis,numberElements,dgdx,d2gdx2,shape,Fract,Mater,...
                    volume,fenerg,geometry,tdisp)gStiffnessPhiPhi4th(PHTelem,sizeBasis,...
                    numberElements,dgdx,d2gdx2,shape,Fract,Mater,volume,fenerg,geometry);

                solver_4th(Input,PF,file_name);
            else

                PF.StiffUU = @(PHTelem,sizeBasis,numberElements,dgdx,shape,Fract,Mater,...
                    volume,tdisp,geometry)gStiffnessUUcubic(PHTelem,sizeBasis,...
                    numberElements,dgdx,shape,Fract,Mater,volume,tdisp,geometry);
                PF.StiffPhiPhi = @(PHTelem,sizeBasis,numberElements,dgdx,d2gdx2,shape,Fract,Mater,...
                    volume,fenerg,geometry,tdisp)gStiffnessPhiPhi4thcubic(PHTelem,sizeBasis,...
                    numberElements,dgdx,d2gdx2,shape,Fract,Mater,volume,fenerg,geometry,tdisp);

                solver_4th(Input,PF,file_name);
            end
        end
        idx = idx+1;
        c2 = clock

        time_hr = c2(4) - c(4)
        time_min = c2(5) - c(5)
        time_sec = c2(6) - c(6)
        TIME = [time_hr; time_min; time_sec]
        save_name = strcat(num2str(idx),'_sim.mat');
        
        save(save_name,'TIME')
        
    end
end