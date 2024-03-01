% >>>> TEST CODE <<<<
proto = HOR; %HOR 29, 70 error
loop = true; sweep = 1; 
prompt = "Continue, press >c<" + newline + "Quite, press >q<"+ newline + "Change sweep number, press >t<"+ newline;

while loop == true
    clc
    figure; hold on; sgtitle("sweep: " + sweep)
    plot(data{proto, ANG}(sweep,:))
    plot(data{proto, FSR}(sweep,:))
    plot([step_index{proto}(sweep,4), step_index{proto}(sweep,4)],[0 5], "LineWidth",2)
    plot([step_index{proto}(sweep,5), step_index{proto}(sweep,5)],[0 5], "LineWidth",2)
    
    correctInput = false; 
    while correctInput == false
        str = input(prompt, 's');
        if strcmp(str,"q")
            disp("Loop stopped")
            loop = false; correctInput = true; 
        elseif strcmp(str,"t")
            sweep = input("New sweep number: ")-1; 
            correctInput = true; 
        elseif strcmp(str,"c") %, sweep == size(data{proto, SOL}))
            correctInput = true; 
        end 

        if correctInput == false
            warning("Input not accepted")
        end
    end
    close all; 
    sweep = sweep + 1;
    if sweep > size(data{proto,SOL},1)
        loop = false; 
    end 
end