package daislab.cspg;

import org.cloudsimplus.vms.Vm;

import java.util.ArrayList;
import java.util.List;

/*
 * Class to calculate the infrastructure cost. We need it to calculate the agent's reward. TODO: I
 * should also extend this to HostCost in order to measure the cost of having many hosts running. So
 * that I can experiment with logics that try to fit as many vms as a host can fit, so that the
 * infrastructure cost does not rise because of many hosts running.
 */
public class VmCost {

    private final double perIterationSmallVmCost;

    private List<Vm> createdVms = new ArrayList<>();
    private double iterationsInHour;
    private SimulationSettings settings;

    public VmCost(final SimulationSettings settings) {
        this.settings = settings;

        // timestepInterval are the seconds we are "staying" at each iteration

        iterationsInHour = 3600 / settings.getTimestepInterval();

        final double perSecondVMCost = settings.getVmHourlyCost() * 0.00028; // 1/3600
        perIterationSmallVmCost = perSecondVMCost * settings.getTimestepInterval();
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
        double totalCost = 0;
        List<Vm> toRemove = new ArrayList<>();
        for (Vm vm : createdVms) {
            // check if the vm is started
            double multiplier = settings.getSizeMultiplier(vm.getDescription());
            final double perIterationVMCost = perIterationSmallVmCost * multiplier;
            if (vm.getStartTime() > -1 && vm.getFinishTime() > -1) {
                // vm was stopped -
                // we continue to pay for it within the last running hour if need to
                if (settings.isPayingForTheFullHour()
                        && (clock <= vm.getFinishTime() + iterationsInHour)) {
                    totalCost += perIterationVMCost;
                } else {
                    toRemove.add(vm);
                }
            } else if (vm.getStartTime() > -1) {
                // vm still running - just add the cost
                totalCost += perIterationVMCost;
            }
        }
        createdVms.removeAll(toRemove);
        return totalCost;
    }

    public void removeVmFromList(final Vm vm) {
        createdVms.remove(vm);
    }
}
