package daislab.cspg;

import org.cloudsimplus.vms.Vm;

import java.util.ArrayList;
import java.util.List;

/*
 * Class to calculate the infrastructure cost.
 * We need it to calculate the agent's reward.
 * TODO: I should also extend this to HostCost
 * in order to measure the cost of having many hosts running.
 * So that I can experiment with logics that try to
 * fit as many vms as a host can fit, so that the
 * infrastructure cost does not rise because of
 * many hosts running.
*/
public class VmCost {

    private final double secondsInIteration;
    private final double perIterationBasicVMCost;
    private final double speedUp;
    
    private List<Vm> createdVms = new ArrayList<>();
    private boolean payForFullHour;
    private double iterationsInHour;

    public VmCost(final double perHourVMCost, final double speedUp, final boolean payForFullHour) {
        this.payForFullHour = payForFullHour;
        this.speedUp = speedUp;
        
        secondsInIteration = speedUp;
        iterationsInHour = 3600 / secondsInIteration;

        final double perSecondVMCost = perHourVMCost * 0.00028; // 1/3600
        perIterationBasicVMCost = perSecondVMCost * secondsInIteration;
    }

    public void addNewVmToList(final Vm vm) {
        createdVms.add(vm);
    }

    public void clear() {
        createdVms.clear();
    }

    // Why calculate this list at each iteration and 
    // do not have simply a counter for each type?
    // Maybe because if payForFullHour is true, we want to keep counting a vm
    // which has been created but stopped.
    public double getVMCostPerIteration(final double clock) {
        double totalCost = 0.0;
        List<Vm> toRemove = new ArrayList<>();
        for(Vm vm : createdVms) {
            // check if the vm is started
            double m = getSizeMultiplier(vm);
            final double perIterationVMCost = perIterationBasicVMCost * m;
            if (vm.getStartTime() > -1) {
                if (vm.getFinishTime() > -1) {
                    // vm was stopped -
                    // we continue to pay for it within the last running hour if need to
                    if (payForFullHour && (clock <= vm.getFinishTime() + iterationsInHour)) {
                        totalCost += perIterationVMCost;
                    } else {
                        toRemove.add(vm);
                    }
                } else {
                    // vm still running - just add the cost
                    totalCost += perIterationVMCost;
                }
            } else {
                // created - not running yet, need to pay for it
                totalCost += perIterationVMCost;
            }
        }
        createdVms.removeAll(toRemove);
        return totalCost;
    }

    public double getSizeMultiplier(final Vm vm) {
        if ("M".equals(vm.getDescription())) {
            return 2.0;
        }
        if ("L".equals(vm.getDescription())) {
            return 4.0;
        }
        return 1.0;
    }

}
