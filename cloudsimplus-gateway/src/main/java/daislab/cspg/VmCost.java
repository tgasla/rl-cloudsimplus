package daislab.cspg;

import org.cloudsimplus.vms.Vm;

import java.util.ArrayList;
import java.util.List;

/*
 * Class to calculate the infrastructure cost.
 * We need it to calculate the agent's reward.
 * TODO: check also the VmCost class of the cloudsimplus
*/
public class VmCost {

    private final double secondsInIteration;
    private final double perIterationSmallVMCost;
    private final double speedUp;
    private List<Vm> createdVms = new ArrayList<>();

    private boolean payForFullHour;
    private double iterationsInHour;

    public VmCost(final double perHourVMCost, final double speedUp, final boolean payForFullHour) {
        this.payForFullHour = payForFullHour;
        this.speedUp = speedUp;
        this.secondsInIteration = this.speedUp;
        this.iterationsInHour = 3600 / this.secondsInIteration;

        final double perSecondVMCost = perHourVMCost * 0.00028; // 1/3600
        this.perIterationSmallVMCost = perSecondVMCost * secondsInIteration;
    }

    public void addNewVmToList(final Vm vm) {
        this.createdVms.add(vm);
    }

    public void clear() {
        createdVms.clear();
    }

    public double getVMCostPerIteration(final double clock) {
        double totalCost = 0.0;
        List<Vm> toRemove = new ArrayList<>();
        for(Vm vm : createdVms) {
            // check if the vm is started
            double m = getSizeMultiplier(vm);
            final double perIterationVMCost = perIterationSmallVMCost * m;
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
