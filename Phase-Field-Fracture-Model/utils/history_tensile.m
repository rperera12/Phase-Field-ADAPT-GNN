function [fenerg]=history_tensile(gaussCord,Fract,numberElements,geometry)

global adder
global cur_angle
nGaussX = geometry.ngaussX;
nGaussY = geometry.ngaussX;
constB = geometry.B;
mgauss = nGaussX*nGaussY;
fenerg = zeros(numberElements,mgauss);



crack_size = 0.1;
crack_loc = adder;
t = cur_angle*pi/180;

x1 = 0; 
y1 = crack_loc;
P1 = [x1,y1];


x2 = P1(1,1)+crack_size*cos(t);
y2 = P1(1,2) + crack_size*sin(t);
P2 = [x2,y2];

disp('P2 is located at ')
disp(P2)

ta = t*180/pi;


for iElem = 1:numberElements
    kgauss = 0;
    for ii=1:nGaussX
        for jj=1:nGaussY
            kgauss = kgauss + 1;

            x3 = gaussCord{iElem}(kgauss,1);
            y3 = gaussCord{iElem}(kgauss,2);
            P3 = [x3,y3];
            x21 = x2 - x1;
            y31 = y3 - y1;
            x31 = x3 - x1;
            y21 = y2 - y1;
            angle_eff = (mod(atan2(x21*y31-x31*y21,x21*x31+y21*y31),2*pi));  % <-- The angle

            x1r = 1; y1r = P1(1,2);
            P1ref = [x1r, y1r];
            x2r = 1; y2r = P2(1,2);
            P2ref = [x2r, y2r];
            
            x12_1 = x1r - x1;
            y32_1 = y3 - y1;
            x32_1 = x3 - x1;
            y12_1 = y1r - y1;
            a1 = (mod(atan2(x12_1*y32_1-x32_1*y12_1,x12_1*x32_1+y12_1*y32_1),2*pi))*360/(2*pi);  % <-- The angle
            if a1 > 180
               a1 = a1-360 ;
            end
            
            
            x12_2 = x2r - x2;
            y32_2 = y3 - y2;
            x32_2 = x3 - x2;
            y12_2 = y2r - y2;
            a2 = (mod(atan2(x12_2*y32_2-x32_2*y12_2,x12_2*x32_2+y12_2*y32_2),2*pi))*360/(2*pi);  % <-- The angle
            if a2 > 180
               a2 = a2-360;
            end
            
            
            
            if cur_angle == 0
                if (gaussCord{iElem}(kgauss,1)>crack_size)
                    dis = sqrt((gaussCord{iElem}(kgauss,1)-crack_size)^2+(gaussCord{iElem}(kgauss,2)-crack_loc)^2);
                    if dis <= Fract.constl/2
                        fenerg(iElem,kgauss) = constB*Fract.cenerg*(1.-dis/(Fract.constl/2))/(2*Fract.constl);
                    end
                elseif (gaussCord{iElem}(kgauss,1)<=crack_size)
                    dis = abs((gaussCord{iElem}(kgauss,2)-crack_loc));
                    if dis <= Fract.constl/2
                        fenerg(iElem,kgauss) = constB*Fract.cenerg*(1-dis/(Fract.constl/2))/(2*Fract.constl);
                    end
                end  
                
            elseif cur_angle > 0
                if (a1 == (-90 + ta)) || (a1 == ta) || (a2 == (-180+ta)) || (a2 == (-90+ta))  
                    v1 = P2 - P1;
                    v2 = P1 - P3;
                    dis = find_perpendicular_to_line(P3, v1, v2, angle_eff);
                    if dis <= Fract.constl/2

                        fenerg(iElem,kgauss) = constB*Fract.cenerg*(1-dis/(Fract.constl/2))/(2*Fract.constl);
                    end

                elseif (a1 == ta) || (a1 == 90) || (a2 == (90+ta)) || (a2 == (-180+ta))  
                    v1 = P2 - P1;
                    v2 = P1 - P3;
                    dis = find_perpendicular_to_line(P3, v1, v2,angle_eff);
                    if dis <= Fract.constl/2

                        fenerg(iElem,kgauss) = constB*Fract.cenerg*(1-dis/(Fract.constl/2))/(2*Fract.constl);
                    end 

                elseif (a1 > (-90 + ta)) && (a1 < ta) && (a2 > (-180+ta)) && (a2 < (-90+ta))   

                    v1 = P2 - P1;
                    v2 = P1 - P3;
                    dis = find_perpendicular_to_line(P3, v1, v2,angle_eff);
                    if dis <= Fract.constl/2

                        fenerg(iElem,kgauss) = constB*Fract.cenerg*(1-dis/(Fract.constl/2))/(2*Fract.constl);
                    end  

                elseif (a1 > ta) && (a1 < 90) && (a2 > (90+ta)) && (a2 < (-180+ta))

                    v1 = P2 - P1;
                    v2 = P1 - P3;
                    dis = find_perpendicular_to_line(P3, v1, v2,angle_eff);
                    if dis <= Fract.constl/2

                        fenerg(iElem,kgauss) = constB*Fract.cenerg*(1-dis/(Fract.constl/2))/(2*Fract.constl);
                    end

                elseif (a1 > -90) && (a1 < (-90+ta)) && (a2 > (-180+ta)) && (a2 < (-90+ta))

                    dis = sqrt( (abs(P3(1) - P1(1)))^2 + (abs(P3(2) - P1(2)))^2 );
                    if dis <= Fract.constl/2
                        fenerg(iElem,kgauss) = constB*Fract.cenerg*(1-dis/(Fract.constl/2))/(2*Fract.constl);
                    end

                else
                   dis = sqrt( (abs(P3(1) - P2(1)))^2 + (abs(P3(2) - P2(2)))^2 );
                   if dis <= Fract.constl/2
                       fenerg(iElem,kgauss) = constB*Fract.cenerg*(1-dis/(Fract.constl/2))/(2*Fract.constl);
                   end
                end
                
            elseif cur_angle <0    
                if (a1 == (ta)) || (a1 == (90+ta)) || (a2 == (90+ta)) || (a2 == (180+ta))  
                    v1 = P2 - P1;
                    v2 = P1 - P3;
                    dis = find_perpendicular_to_line(P3, v1, v2, angle_eff);
                    if dis <= Fract.constl/2

                        fenerg(iElem,kgauss) = constB*Fract.cenerg*(1-dis/(Fract.constl/2))/(2*Fract.constl);
                    end

                elseif (a1 == (-90)) || (a1 == ta) || (a2 == (180+ta)) || (a2 == (-90+ta))  
                    v1 = P2 - P1;
                    v2 = P1 - P3;
                    dis = find_perpendicular_to_line(P3, v1, v2,angle_eff);
                    if dis <= Fract.constl/2

                        fenerg(iElem,kgauss) = constB*Fract.cenerg*(1-dis/(Fract.constl/2))/(2*Fract.constl);
                    end 

                elseif (a1 > (ta)) && (a1 < 90+ta) && (a2 > (90+ta)) && (a2 < (180+ta))   

                    v1 = P2 - P1;
                    v2 = P1 - P3;
                    dis = find_perpendicular_to_line(P3, v1, v2,angle_eff);
                    if dis <= Fract.constl/2

                        fenerg(iElem,kgauss) = constB*Fract.cenerg*(1-dis/(Fract.constl/2))/(2*Fract.constl);
                    end  

                elseif (a1 > -90) && (a1 < ta) && (a2 > (180+ta)) && (a2 < (-90+ta))

                    v1 = P2 - P1;
                    v2 = P1 - P3;
                    dis = find_perpendicular_to_line(P3, v1, v2,angle_eff);
                    if dis <= Fract.constl/2

                        fenerg(iElem,kgauss) = constB*Fract.cenerg*(1-dis/(Fract.constl/2))/(2*Fract.constl);
                    end

                elseif (a1 > (90+ta)) && (a1 < (90)) && (a2 > (90+ta)) && (a2 < (180+ta))

                    dis = sqrt( (abs(P3(1) - P1(1)))^2 + (abs(P3(2) - P1(2)))^2 );
                    if dis <= Fract.constl/2
                        fenerg(iElem,kgauss) = constB*Fract.cenerg*(1-dis/(Fract.constl/2))/(2*Fract.constl);
                    end

                else
                   dis = sqrt( (abs(P3(1) - P2(1)))^2 + (abs(P3(2) - P2(2)))^2 );
                   if dis <= Fract.constl/2
                       fenerg(iElem,kgauss) = constB*Fract.cenerg*(1-dis/(Fract.constl/2))/(2*Fract.constl);
                   end
                end
                
            end            
        end
    end
end


% end
end

function perp = find_perpendicular_to_line(pt, v1, v2,angle_eff)
% Compute distance

Ax = v1(1);
Ay = v1(2);
Bx = v2(1);
By = v2(2);


cross_val = Ax.*By-Ay.*Bx;
d = abs(cross_val) / norm(v1);




% Perpendicular is its base vector times length
perp =d;

end
